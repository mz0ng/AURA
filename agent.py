import numpy as np

class QLearningAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        learning_rate=0.015, 
        discount_factor=0.9, 
        epsilon=1.0, 
        epsilon_decay=0.99, 
        epsilon_min=0.01,
        initial_trust=0.1,
        table=''
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_trust = initial_trust
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        # self.Q = np.zeros((self.state_size, self.action_size))
        self.Q = np.load(table)
        self.hist = []
        self.trust_scores = {}
        
    def choose_action(self, state):
        # Epsilon-greedy action selection
        # if np.random.rand() < self.epsilon:
        #     return np.random.randint(self.action_size)
        # else:
        #     return np.argmax(self.Q[state, :])
        return np.argmax(self.Q[state, :])
    
    def update_trust(self, state, rl_action, reward, llm_action):
        # Initialize for new states
        if state not in self.trust_scores:
            self.trust_scores[state] = self.initial_trust

        alpha = 0.01  # small learning rate
        if rl_action == llm_action:
            # Weighted by sign and magnitude of reward
            self.trust_scores[state] += alpha * reward  
        else:
            self.trust_scores[state] -= alpha * reward  
        self.trust_scores[state] = np.clip(self.trust_scores[state], 0, 1)
        
    def make_decision(self, state, llm_action, rl_action):
        trust = self.trust_scores.get(state, self.initial_trust)
        if np.random.rand() < trust:
            return llm_action
        return rl_action
    
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state, :])
        td_target = reward + (self.gamma * self.Q[next_state, best_next_action])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def add_pair(self, state, action):
        self.hist.append((state, action))

    def apply_delayed_reward(self, d_reward):
        for state, action in self.hist:
            self.Q[state, action] += self.lr * d_reward
        self.hist = []

    def update_params(self):
        pass