import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium.wrappers import record_video
import numpy as np
import torch

from pettingzoo.mpe import simple_tag_v3

from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from supersuit import pad_observations_v0
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


import warnings

warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--gamma", type=float, default=0.75, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=100)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 24)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    # parser.add_argument(
    #     "--resume-path",
    #     type=str,
    #     default="",
    #     help="the path of agent pth file " "for resuming from a pre-trained agent",
    # )
    # parser.add_argument(
    #     "--opponent-path",
    #     type=str,
    #     default="",
    #     help="the path of opponent agent pth file "
    #     "for resuming from a pre-trained agent",
    # )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_prey: Optional[BasePolicy] = None,
    agent_predator1: Optional[BasePolicy] = None,
    agent_predator2: Optional[BasePolicy] = None,
    agent_predator3: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:

    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # if agent_learn is None:

    # DQN model for Prey
    prey_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)

    if optim is None:
        _optim = torch.optim.Adam(prey_net.parameters(), lr=args.lr)

    agent_prey = DQNPolicy(
        model=prey_net,
        optim=_optim,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    # if args.resume_path:
    #     agent_learn.load_state_dict(torch.load(args.resume_path))

    # if agent_opponent is None:
    # DQN model for Predator
    predator_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)

    if optim is None:
        _optim = torch.optim.Adam(predator_net.parameters(), lr=args.lr)

        # if args.opponent_path:
        #     agent_opponent = deepcopy(agent_learn)
        #     agent_opponent.load_state_dict(torch.load(args.opponent_path))
        # else:

    predator_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)

    # agent_predator = RandomPolicy()
    agent_predator1 = DQNPolicy(
        model=predator_net,
        optim=_optim,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    agent_predator2 = DQNPolicy(
        model=predator_net,
        optim=_optim,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )
    agent_predator3 = DQNPolicy(
        model=predator_net,
        optim=_optim,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    agents = [agent_predator3, agent_predator2, agent_predator1, agent_prey]

    policy = MultiAgentPolicyManager(agents, env)
    return policy, _optim, env.agents


def get_env(render_mode=None):
    return PettingZooEnv(
        pad_observations_v0(
            simple_tag_v3.env(
                num_good=1,
                num_adversaries=3,
                num_obstacles=1,
                max_cycles=25,
                continuous_actions=False,
                render_mode=render_mode,
            )
        )
    )


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_prey: Optional[BasePolicy] = None,
    agent_predator1: Optional[BasePolicy] = None,
    agent_predator2: Optional[BasePolicy] = None,
    agent_predator3: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy, BasePolicy, BasePolicy, BasePolicy]:

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args,
        agent_prey=agent_prey,
        agent_predator1=agent_predator1,
        agent_predator2=agent_predator2,
        agent_predator3=agent_predator3,
        optim=optim,
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "simple_tag_v3", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "simple_tag_v3", "dqn", "policy.pth"
            )
        # torch.save(policy.policies[agents[1]].state_dict(), model_save_path)
        for i in range(4):
            torch.save(
                policy.policies[agents[i]].state_dict(), f"{model_save_path}_{i}"
            )

    def stop_fn(mean_rewards):
        return mean_rewards >= 100#args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_train)
        policy.policies[agents[1]].set_eps(args.eps_train)
        policy.policies[agents[2]].set_eps(args.eps_train)
        policy.policies[agents[3]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(args.eps_test)
        policy.policies[agents[1]].set_eps(args.eps_test)
        policy.policies[agents[2]].set_eps(args.eps_test)
        policy.policies[agents[3]].set_eps(args.eps_test)

    def reward_metric(rews):
        return np.mean(rews, axis=1)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return (
        result,
        policy.policies[agents[3]],
        policy.policies[agents[2]],
        policy.policies[agents[1]],
        policy.policies[agents[0]],
    )


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_prey: Optional[BasePolicy] = None,
    agent_predator1: Optional[BasePolicy] = None,
    agent_predator2: Optional[BasePolicy] = None,
    agent_predator3: Optional[BasePolicy] = None,
) -> None:
    env = get_env(render_mode="human")
    env = DummyVectorEnv([lambda: env])
    policy, optim, agents = get_agents(
        args,
        agent_prey=agent_prey,
        agent_predator1=agent_predator1,
        agent_predator2=agent_predator2,
        agent_predator3=agent_predator3,
    )
    policy.eval()
    policy.policies[agents[0]].set_eps(args.eps_test)
    policy.policies[agents[1]].set_eps(args.eps_test)
    policy.policies[agents[2]].set_eps(args.eps_test)
    policy.policies[agents[3]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    # print("debug @ rews: ", rews, rews.shape)
    print(f"Final reward: {np.mean(rews, axis=1)}, length: {lens.mean()}")


# train the agent and watch its performance in a match!
args = get_args()
result, prey_policy, predator_policy1, predator_policy2, predator_policy3 = train_agent(
    args
)
watch(
    args,
    agent_prey=prey_policy,
    agent_predator1=predator_policy1,
    agent_predator2=predator_policy2,
    agent_predator3=predator_policy3,
)
