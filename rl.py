import numpy as np
import pandas as pd
import random
import pickle
import ast
import matplotlib.pyplot as plt
from env import CellTowerEnv
from agent import SingleAgent

# power level to capacity mappings
MAPPING1 = {
        1: 25,
        2: 35,
        3: 43,
        4: 50
    }
MAPPING2 = {
        1: 8,
        2: 11,
        3: 14,
        4: 17,
        5: 21,
        6: 25,
        7: 28,
        8: 30
    }


def train_q_learning(episodes, df1, df2):
    env1 = CellTowerEnv(MAPPING1, 50)
    env2 = CellTowerEnv(MAPPING2, 30)

    agent1 = SingleAgent(
        num_power_level=env1.max_power,
        action_size=env1.action_space_size,
        learning_rate=0.7,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        table=''#'ag1_qtable.npy'
    )
    agent2 = SingleAgent(
        num_power_level=env2.max_power,
        action_size=env2.action_space_size,
        learning_rate=0.7,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        table=''#'ag2_qtable.npy'
    )
    
    rewards_hist1 = []
    rewards_hist2 = []
    ag1_dropped = []
    ag2_dropped = []

    max_steps = len(df1)
    
    for ep in range(episodes):
        print(ep)
        env1.reset()
        env2.reset()
        total1 = 0
        total2 = 0
        d1 = 0
        d2 = 0

        for t in range(max_steps):
            calls1 = df1.loc[t, 'calls']
            calls2 = df2.loc[t, 'calls']

            # --- stations handle their own calls ---
            state1 = env1.get_state()
            action1 = agent1.choose_action(state1)
            handled1, dropped1= env1.apply_action(action1, calls1)

            state2 = env2.get_state()
            action2 = agent2.choose_action(state2)
            handled2, dropped2= env2.apply_action(action2, calls2)

            # --- stations handle handoffs ---
            if dropped1 >= 0 and action1[1] == 1:
                added2, failed2 = env2.add_requests(calls1[handled1:])
            else:
                added2 = 0
                failed2 = 0
            if dropped2 >= 0 and action2[1] == 1:
                added1, failed1 = env1.add_requests(calls2[handled2:])
            else:
                added1 = 0
                failed1 = 0
            # --- update how many requests got handled and dropped ---
            handled1 += added1
            dropped2 -= added1
            handled2 += added2
            dropped1 -= added2

            d1 += dropped1
            d2 += dropped2

            # --- compute rewards ---
            reward1 = env1.compute_reward(action1, handled1, dropped1)
            reward2 = env2.compute_reward(action2, handled2, dropped2)

            total1 += reward1
            total2 += reward2

            # --- Q-learning update ---
            agent1.learn(state1, action1, reward1, env1.get_state())
            agent2.learn(state2, action2, reward2, env2.get_state())

        # decay epsilon
        agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay, agent1.epsilon_min)
        agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay, agent2.epsilon_min)
        
        rewards_hist1.append(total1)
        rewards_hist2.append(total2)

        ag1_dropped.append(d1)
        ag2_dropped.append(d2)

    l = range(len(rewards_hist1))
    # Plot both lists against indices
    plt.plot(l, rewards_hist1, label='Agent1')
    plt.plot(l, rewards_hist2, label='Agent2')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    return rewards_hist1, rewards_hist2

# Run training
if __name__ == "__main__":
    df1 = pd.read_csv('data/ntraffic_station1.csv')
    df1['calls'] = df1['calls'].apply(ast.literal_eval)
    df2 = pd.read_csv('data/ntraffic_station2.csv')
    df2['calls'] = df2['calls'].apply(ast.literal_eval)

    r1, r2= train_q_learning(1500, df1.iloc[:800], df2.iloc[:800])

