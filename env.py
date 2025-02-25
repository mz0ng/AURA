import random

class CellTowerEnv:
    def __init__(self, min_power, max_power, min_dist, max_dist, user):
        self.max_power = max_power
        self.min_power = min_power
        self.power_level = 0
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.max_user = user
        self.user_attached = 0
        self.num_power_levels = max_power - min_power + 1
        self.signals = []
        self.snrs = []
        # coverage quality: good, fair, poor
        self.num_cover_states = 3
        # reached max_users or not
        self.num_user_states = 2
        self.observation_space_size = self.num_power_levels * self.num_cover_states * self.num_user_states
        # 4 actions: increase power, decrease power, do nothing, handoff last request
        self.action_space_size = 4

        self.reset()

    def reset(self):
        # random initialization
        self.power_level = random.randint(self.min_power+3, self.max_power)
        self.user_attached = random.randint(self.max_user-3, self.max_user)
        self.signals = [random.randint(-120, -50) for _ in range(self.user_attached)]
        self.snrs = [random.randint(10, 30) for _ in range(self.user_attached)]
        return self._get_state()
    
    def _get_state(self):
        # check capacity
        if self.user_attached >= self.max_user:
            user_idx = 1
        else:
            user_idx = 0
        # check coverage state
        signal_frac = sum(1 for x in self.signals if x < -100)/self.user_attached if self.user_attached != 0 else 0
        snr_frac = sum(1 for x in self.snrs if x < 15)/self.user_attached if self.user_attached != 0 else 0
        # greate portion of poor signal and snr ---> poor coverage quality
        if signal_frac > 0.5 or snr_frac > 0.5:
            cover_idx = 2
        # small portion of poor signal and snr ---> good coverage quality
        elif signal_frac <= 0.2 and snr_frac <= 0.2:
            cover_idx = 0
        else:
            cover_idx = 1
        # get power level
        power_idx = self.power_level - self.min_power
        # compute state index
        state_idx = power_idx * (self.num_cover_states * self.num_user_states) + (cover_idx * self.num_user_states) + user_idx
        return state_idx, power_idx, cover_idx, user_idx, 1-signal_frac, 1-snr_frac
    
    def get_values(self):
        signal_frac = sum(1 for x in self.signals if x >= -100)/self.user_attached if self.user_attached != 0 else 0
        snr_frac = sum(1 for x in self.snrs if x >= 15)/self.user_attached if self.user_attached != 0 else 0
        return signal_frac, snr_frac
    
    def add_request(self, signal, snr):
        if self.user_attached >= self.max_user:
            return False
        self.user_attached += 1
        self.signals.append(signal)
        self.snrs.append(snr)
        return True

    def change_env(self):
        add = random.choice([True, False])
        if add == True or self.user_attached == 0:
            self.user_attached += 1
            self.signals.append(random.randint(-120, -50))
            self.snrs.append(random.randint(10, 30))
        else:
            idx = random.randint(0, self.user_attached-1)
            self.signals.pop(idx)
            self.snrs.pop(idx)
            self.user_attached -= 1
        return self._get_state()
    
    def step(self, action):
        signal = None
        snr = None
        _, power_idx, cover_idx, user_idx, _, _ = self._get_state()
        
        # compute reward for the action
        action_reward = 0
        # action = increase power
        if action == 0:
            if self.power_level >= self.max_power:
                action_reward += -3
            else:
                # poor coverage quality
                if cover_idx == 2:
                    action_reward += 2
                elif cover_idx == 0:
                    action_reward += -1
                self.power_level += 1
        # action: decrease power
        elif action == 1:
            if self.power_level <= self.min_power:
                action_reward += -3
            else:
                # poor coverage quality
                if cover_idx == 2:
                    action_reward += -2
                elif cover_idx == 0:
                    action_reward += 1
                self.power_level -= 1
        # do nothing
        elif action == 2:
            if self.min_power < self.power_level < self.max_power:
                if cover_idx == 2 or cover_idx == 1:
                    action_reward += -1
                else:
                    action_reward += 2
        # handoff one request 
        else:
            # if max users attached
            if user_idx == 1:
                self.user_attached -= 1
                signal = self.signals.pop()
                snr = self.snrs.pop()
                action_reward += 2
            else:
                action_reward += -1
        
        # power efficiency
        power_reward = 1 - ((power_idx / self.num_power_levels) ** 1.5)

        # connection
        conn_reward = 0
        if cover_idx == 2:
            conn_reward = -1
        elif cover_idx == 0:
            conn_reward = 1

        # weighted total reward
        total_reward = action_reward*1.1 + power_reward*2.3 + conn_reward
        
        # get the next state
        next_state, _, _, _, _, _ = self._get_state()
        
        return next_state, total_reward, signal, snr
    