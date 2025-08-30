import numpy as np
import itertools
import random
from collections import defaultdict

class SingleAgent:
    def __init__(
        self, 
        num_power_level, 
        action_size, 
        learning_rate, 
        discount_factor, 
        epsilon, 
        epsilon_decay, 
        epsilon_min,
        table='',
        initial_trust=0.5
        
    ):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_trust = initial_trust
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.states = list(itertools.product(list(range(1, num_power_level+1)), [0, 1, 2], [0, 1]))
        self.actions = list(itertools.product(list(range(1, num_power_level+1)), [0, 1]))
        if table != '':
            self.Q = np.load(table, allow_pickle=True).item()
        else:
            self.Q = defaultdict(float)

        # self.Q = np.load(table)
        self.trust_scores = defaultdict(float)
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            values = np.array([self.Q[(state, a)] for a in self.actions], dtype=float)
            max_q = values.max()
            best_idxs = np.flatnonzero(values == max_q)
            return self.actions[random.choice(best_idxs.tolist())]
    
    def learn(self, s, a, r, next_state):
        q_sa = self.Q[(s, a)]
        next_max = max(self.Q[(next_state, a_p)] for a_p in self.actions)
        self.Q[(s, a)] = q_sa + self.lr * (r + self.gamma * next_max - q_sa)
    
    def update_trust(self, state, decision, reward, llm_action, rl_action):
        expectation = self.Q[(state, rl_action)]
        old = self.trust_scores.get(state, self.initial_trust)
        if decision == llm_action:
            if llm_action != rl_action:
                success = 1.0 if reward >= expectation else 0.0
                self.trust_scores[state] = 0.95 * old + 0.05* success
        elif decision != llm_action:
            if state[3] > 0 or reward < 0:
                self.trust_scores[state] = 1.05 * old
            else:
                self.trust_scores[state] = 0.999 * old
        return (old, self.trust_scores[state])
        
    def make_decision(self, state, llm_action, rl_action):
        if rl_action == llm_action:
            return rl_action
        p = float(self.trust_scores.get(state, self.initial_trust))
        p = max(0.0, min(1.0, p))
        return llm_action if random.random() < p else rl_action

    def add_pair(self, state, action):
        self.hist.append((state, action))

    def apply_delayed_reward(self, hist, d_rewards, rate):
        for i, r in enumerate(d_rewards):
            s = hist[i]['state']
            a = hist[i]['action']
            q_sa = self.Q[(s, a)]
            self.Q[(s, a)] = q_sa + rate * r

    def update_params(self):
        pass