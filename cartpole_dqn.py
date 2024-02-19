from pettingzoo.mpe import simple_tag_v3
import tianshou as ts
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import torch, numpy as np
from torch import nn
import random
import copy
import sys
import envpool

import warnings
warnings.filterwarnings('ignore')


# train_env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=2500, continuous_actions=False, render_mode=None)
# test_env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=2500, continuous_actions=False, render_mode="human")

env = gym.make('CartPole-v1',render_mode='human')

train_env = envpool.make_gymnasium("CartPole-v1", num_envs=10)
test_env = envpool.make_gymnasium("CartPole-v1", num_envs=100)

# train_env = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(10)])
# test_env = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])


# train_env.reset(seed=42)
# test_env.reset(seed=42)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.n or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    action_space= env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)

train_collector = ts.data.Collector(policy, train_env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_env, exploration_noise=True)

# result = self.policy(self.data, last_state)                         # the agent predicts the batch action from batch observation
# act = to_numpy(result.act)
# self.data.update(act=act)                                           # update the data with new action/policy
# result = self.env.step(act, ready_env_ids)                          # apply action to environment
# obs_next, rew, done, info = result
# self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)  # update the data with new state/reward/done/info

# result = ts.trainer.OffpolicyTrainer(
#     policy=policy,
#     train_collector=train_collector,
#     test_collector=test_collector,
#     max_epoch=10, step_per_epoch=10000, step_per_collect=10,
#     update_per_step=0.1, episode_per_test=100, batch_size=64,
#     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
#     test_fn=lambda epoch, env_step: policy.set_eps(0.05),
#     stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
# ).run()
# print(f'Finished training! Use {result["duration"]}')



writer = SummaryWriter('log/dqn')
logger = TensorboardLogger(writer)

# torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)



def env_logger(env, verbose = True):
    sys.stdout = open('simple_tag_log.txt', 'w')
    print("state_space = ", env.state_space)
    print("observation_space = ", env.observation_spaces)
    print("action_space = ", env.action_spaces)

    print("======================================")
    print()
    for agent in env.agent_iter():
        print("Agent=",agent)
        observation, reward, termination, truncation, info = env.last()
        print("Observation : ",observation)
        print("Reward:",reward)
        print("Termination: ",termination)
        print("Truncation",truncation)
        print("Selected Agent: ",env.agent_selection)
        print("Agent Observation Space: ",env.observation_space(agent))
        print("Agent Action Space: ",env.action_space(agent))

        print("************************")
        print()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            print("action = ",action)
        env.step(action)
    
    print("state_space = ", env.state_space)
    print("observation_space = ", env.observation_spaces)
    print("action_space = ", env.action_spaces)

    print("======================================")

    env.close()
    sys.stdout.close()

# env_logger(train_env)













# Q_table = {'adversary_0': np.zeros((16,5)), 'adversary_1': np.zeros((16,5)), 'adversary_2': np.zeros((16,5)), 'agent_0': np.zeros((14,5))}
# learning_rate = 0.1
# discount_rate = 0.9

# for agent in env.agent_iter():


#     observation, reward, termination, truncation, info = env.last()
#     # break


#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy

#         # Q-learning algorithm
#         # Q_table[agent][state,action] = Q_table[agent][state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
#         action = env.action_space(agent).sample()

#     env.step(action)
# env.close()



# # # If want to see demo of taxi, Put RENDER = 1
# # RENDER = 1   
# # def QLearning_train(epsilon,learning_rate,discount_rate,ep):
# #     # create tag environment
# #     env_train = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False, render_mode="human")

# #     state_size = env_train.observation_space.n  # 500 state
# #     action_size = env_train.action_space.n      # 6 actions
    
# #     # initialize q-table 
# #     qtable = np.zeros((state_size, action_size))
    
# #     # training variables
# #     num_episodes = ep
# #     max_steps = 99  # per episode
# #     episode = 1
    
# #     # training
# #     for i in range(num_episodes):
# #         # reset the environment
# #         state, _ = env_train.reset(seed=42)
# #         done = False
# #         q = copy.deepcopy(qtable)
# #         for s in range(max_steps):  
# #             # exploration-exploitation tradeoff
# #             if random.uniform(0, 1) < epsilon:                      # Epsilon Greedy
# #                 # explore
# #                 action = env_train.action_space.sample()
# #             else:
# #                 # exploit
# #                 action = np.argmax(qtable[state, :]) # this cannot be same as greedy policy!

# #             # take action and observe reward
# #             new_state, reward, done, info, _ = env_train.step(action)
            

# #             # Q-learning algorithm
# #             qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            
                      
# #             # Update to our new state
# #             state = new_state

# #             # if done, finish episode
# #             if done==True:
# #                 break

# #         episode+=1
    
# #     env_train.close()
    
# #     print(f"Training completed over {episode} episodes")
# #     return qtable
    
# # def QLearning_test(qtable,no_of_demo):
# #     if RENDER == 1:
# #         env_test = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False, render_mode="human")

# #     else: 
# #         env_test = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)

    
# #     max_steps = 99
    
# #     score_lst = []
# #     steps_lst = []
    
# #     for i in range(no_of_demo):
# #         state, _ = env_test.reset()
# #         done = False
# #         rewards = 0

# #         for s in range(max_steps):
# #             # print(f"TRAINED AGENT")
# #             # print("Step {}".format(s + 1))

# #             action = np.argmax(qtable[state, :])
# #             new_state, reward, done, info, _ = env_test.step(action)
# #             rewards += reward
            
# #             # env_test.render()
# #             # print(f"score: {rewards}")
# #             state = new_state

# #             if done == True:
# #                 break

# #             for agent in env.agent_iter():
# #                 observation, reward, termination, truncation, info = env.last()

# #                 if termination or truncation:
# #                     action = None
# #                 else:
# #                     # this is where you would insert your policy
# #                     action = env.action_space(agent).sample()

# #                 env.step(action)
 
# #         score_lst.append(rewards)
# #         steps_lst.append(s)
    
 
# #     env_test.close()
    
# #     return score_lst, steps_lst

# # def QLearning(epsilon,learning_rate,discount_rate,ep,no_of_demo):
# #     Q_table = QLearning_train(epsilon,learning_rate,discount_rate,ep)
# #     rewards, steps = QLearning_test(Q_table,no_of_demo)
# #     avg_reward = np.mean(rewards) 
# #     avg_steps = np.mean(steps)
# #     std_reward = np.std(rewards)
# #     std_steps = np.std(steps)
    
# #     return avg_reward,std_reward,avg_steps,std_steps

# # if __name__ == "__main__":
# #     QLearning(epsilon=0.5,learning_rate=0.9,discount_rate=0.9,ep=1000,no_of_demo=10)