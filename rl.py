import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
from env import CellTowerEnv


# class CellTowerEnv:
#     def __init__(self, min_power, max_power, min_dist, max_dist, user):
#         self.max_power = max_power
#         self.min_power = min_power
#         self.max_dist = max_dist
#         self.min_dist = min_dist
#         self.max_user = user
#         self.num_power_levels = max_power - min_power + 1
#         # coverage quality: good, fair, poor
#         self.num_cover_states = 3
#         # reached max_users or not
#         self.num_user_states = 2
#         self.observation_space_size = self.num_power_levels * self.num_cover_states * self.num_user_states
#         # 4 actions: increase power, decrease power, do nothing, handoff last request
#         self.action_space_size = 4

#         self.reset()

#     def reset(self):
#         # random initialization
#         self.power_level = random.randint(self.min_power, self.max_power)
#         self.user_attached = random.randint(0, self.max_user)
#         self.signals = [random.randint(-120, -50) for _ in range(self.user_attached)]
#         self.snrs = [random.randint(10, 30) for _ in range(self.user_attached)]
#         return self._get_state()
    
#     def _get_state(self):
#         # check capacity
#         if self.user_attached >= self.max_user:
#             user_idx = 1
#         else:
#             user_idx = 0
#         # check coverage state
#         signal_frac = sum(1 for x in self.signals if x <= -90)/self.user_attached if self.user_attached != 0 else 0
#         snr_frac = sum(1 for x in self.snrs if x <= 14)/self.user_attached if self.user_attached != 0 else 0
#         # greate portion of poor signal or snr ---> poor coverage quality
#         if signal_frac > 0.3 or snr_frac > 0.3:
#             cover_idx = 2
#         # small portion of poor signal or snr ---> good coverage quality
#         elif signal_frac < 0.15 and snr_frac < 0.15:
#             cover_idx = 0
#         else:
#             cover_idx = 1
#         # get power level
#         power_idx = self.power_level - self.min_power
#         # compute state index
#         state_idx = power_idx * (self.num_cover_states * self.num_user_states) + (cover_idx * self.num_user_states) + user_idx
#         return state_idx, power_idx, cover_idx, user_idx
    
#     def update(self, signal, snr, add):
#         if add == True:
#             self.user_attached += 1
#             self.signals.append(signal)
#             self.snrs.append(snr)
#         else:
#             self.user_attached -= 1
#             self.signals.remove(signal)
#             self.snrs.remove(snr)
#         return self._get_state()
    
#     def step(self, action):
#         _, power_idx, cover_idx, user_idx = self._get_state()
        
#         # compute reward for the action
#         action_reward = 0
#         # action = increase power
#         if action == 0:
#             if self.power_level >= self.max_power:
#                 action_reward = -2
#             else:
#                 # poor coverage quality
#                 if cover_idx == 2:
#                     action_reward = 2
#                 self.power_level += 1
#         # action: decrease power
#         elif action == 1:
#             if self.power_level <= self.min_power:
#                 action_reward = -2
#             else:
#                 # poor coverage quality
#                 if cover_idx == 2:
#                     action_reward = -2
#                 self.power_level -= 1
#         # do nothing - passed
#         # handoff one request 
#         elif action == 3:
#             # if max users attached
#             if user_idx == 1:
#                 self.user_attached -= 1
#                 self.signals.pop(0)
#                 self.snrs.pop(0)
#                 action_reward = 2
#             else:
#                 action_reward = -1
        
#         # power efficiency
#         power_reward = 1 - (power_idx/self.num_power_levels)

#         # weighted total reward
#         total_reward = action_reward*1.5 + power_reward*3
        
#         # get the next state
#         next_state, _, _, _ = self._get_state()
        
#         return next_state, total_reward


class QLearningAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        learning_rate=0.1, 
        discount_factor=0.9, 
        epsilon=1.0, 
        epsilon_decay=0.99, 
        epsilon_min=0.01, 
        table=''
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        # self.Q = np.zeros((self.state_size, self.action_size))
        self.Q = np.load(table)
        
    def choose_action(self, state):
        # Epsilon-greedy action selection
        # if np.random.rand() < self.epsilon:
        #     return np.random.randint(self.action_size)
        # else:
        #     return np.argmax(self.Q[state, :])
        return np.argmax(self.Q[state, :])
    
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + (self.gamma * self.Q[next_state, best_next_action])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def update_params(self):
        pass


def train_q_learning(num_episodes, max_steps):
    env1 = CellTowerEnv(43, 46, 1.6, 5, 50)
    env2 = CellTowerEnv(30, 37, 0.1, 1.3, 30)
    env1_vals = []
    env2_vals = []
    
    agent1 = QLearningAgent(
        state_size=env1.observation_space_size,
        action_size=env1.action_space_size,
        learning_rate=0.05,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        table='ag1_qtable.npy'
    )
    agent2 = QLearningAgent(
        state_size=env2.observation_space_size,
        action_size=env2.action_space_size,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        table='ag2_qtable.npy'
    )
    
    rewards_hist1 = []
    rewards_hist2 = []
    
    for episode in range(num_episodes):
        ag1_state = list(env1.reset())
        total_reward1 = 0
        ag2_state = list(env2.reset())
        total_reward2 = 0
        ag1_failed = 0
        ag2_failed = 0

        
        for step in range(max_steps):
            action_r1 = agent1.choose_action(ag1_state[0])
            next_r1, reward1, sig1, snr1 = env1.step(action_r1)
            agent1.learn(ag1_state[0], action_r1, reward1, next_r1)
            total_reward1 += reward1
            # if agent choose to handoff, hand over to the other agent
            if sig1 is not None:
                result1 = env2.add_request(sig1, snr1)
                if result1 == False:
                    ag1_failed += 1


            action_r2 = agent2.choose_action(ag2_state[0])
            next_r2, reward2, sig2, snr2 = env2.step(action_r2)
            agent2.learn(ag2_state[0], action_r2, reward2, next_r2)
            total_reward2 += reward2
            # if agent choose to handoff, hand over to the other agent
            if sig2 is not None:
                result2 = env1.add_request(sig2, snr2)
                if result2 == False:
                    ag2_failed += 1

            # simulate change in environment
            if step != (max_steps-1):
                ag1_state = list(env1.change_env())
                ag2_state = list(env2.change_env())
        print(ag1_failed, ag2_failed)


        # # decay epsilon
        # agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay, agent1.epsilon_min)
        # agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay, agent2.epsilon_min)
        
        rewards_hist1.append(total_reward1)
        rewards_hist2.append(total_reward2)

        env1_vals.append([ag1_state[1]]+list(env1.get_values())+[ag1_failed])
        env2_vals.append([ag2_state[1]]+list(env2.get_values())+[ag2_failed])

        # if (episode+1) % 100 == 0:
        #     print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent1.epsilon:.2f}")
    
    # np.save('ag1_qtable.npy', agent1.Q)
    # np.save('ag2_qtable.npy', agent2.Q)
    return agent1, rewards_hist1, env1_vals, agent2, rewards_hist2, env2_vals

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Run training
if __name__ == "__main__":
    # df = pd.read_csv('data.csv')
    ag1, r1, v1, ag2, r2, v2 = train_q_learning(100, 10)
    # print(r1)
    # print(sum(r1)/len(r1))
    # print(sum(r2)/len(r2))
    with open('rl_ag1.pkl', 'wb') as f1:
        pickle.dump(v1, f1)
    with open('rl_ag2.pkl', 'wb') as f2:
        pickle.dump(v2, f2)
    # smooth_r1 = moving_average([x for x, y, z, k in v2])
    # episodes = list(range(1, len(smooth_r1) + 1))
    # plt.plot(episodes, smooth_r1)
    # plt.grid(True)
    # plt.show()
    # loaded_q_table = np.load('ag1_qtable.npy')
    # print("Loaded Q-table:\n", loaded_q_table)
