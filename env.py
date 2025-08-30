import random

class CellTowerEnv:
    def __init__(self, mapping, user):
        self.mapping = mapping
        self.max_power = list(mapping)[-1]
        self.min_power = 1
        self.power_level = 1
        self.num_power_levels = self.max_power - self.min_power + 1
        self.max_user = user
        self.user_attached = 0
        self.requests = []
        # coverage quality: good, fair, poor
        self.num_cover_states = 3
        # reached max_users or not
        self.num_user_states = 2
        # action pair: (power_level, if_handoff_additional_load)
        self.action_space_size = self.num_power_levels * 2
        self.last_rejected = 0
        self.reset()

    def reset(self):
        # random initialization
        self.power_level = random.randint(self.min_power, self.max_power)
        self.user_attached = random.randint(self.max_user, self.max_user)
        signals = [random.randint(-120, -50) for _ in range(self.user_attached)]
        snrs = [random.randint(0, 30) for _ in range(self.user_attached)]
        self.requests = list(zip(signals, snrs))
        return self.get_state()
    
    def get_state(self):
        # check capacity
        if self.user_attached >= self.mapping[self.power_level]:
            user_idx = 1
        else:
            user_idx = 0
        # check coverage state
        f, s = zip(*self.requests)
        signals = list(f)
        snrs = list(s)
        signal_frac = sum(1 for x in signals if x < -90)/self.user_attached if self.user_attached != 0 else 0
        snr_frac = sum(1 for x in snrs if x < 20)/self.user_attached if self.user_attached != 0 else 0
        # great portion of poor signal and snr ---> poor coverage quality
        if signal_frac > 0.5 or snr_frac > 0.5:
            cover_idx = 2
        # small portion of poor signal and snr ---> good coverage quality
        elif signal_frac <= 0.2 and snr_frac <= 0.2:
            cover_idx = 0
        else:
            cover_idx = 1
        return (self.power_level, cover_idx, user_idx, self.last_rejected)
    
    def get_values(self):
        f, s = zip(*self.requests)
        signals = list(f)
        snrs = list(s)
        signal_frac = sum(1 for x in signals if x >= -90)/self.user_attached if self.user_attached != 0 else 0
        snr_frac = sum(1 for x in snrs if x >= 20)/self.user_attached if self.user_attached != 0 else 0
        return signal_frac, snr_frac
    
    def add_requests(self, calls):
        if self.user_attached >= self.mapping[self.power_level]:
            return 0, len(calls)
        availability = self.mapping[self.power_level] - self.user_attached
        availability = min(availability, len(calls))
        self.requests += calls[:availability]
        return availability, len(calls)-availability
    
    def apply_action(self, action, calls):
        self.user_attached = 0
        self.power_level, handoff = action
        handled, dropped = self.add_requests(calls)
        return handled, dropped
    
    def compute_reward(self, action, handled, dropped):
        signal_frac, snr_frac = self.get_values()

        reward = (0.15 * handled + 5 * signal_frac + 5 * snr_frac)
        # small penalty for power inefficiency
        if self.power_level > 1 and self.user_attached <= self.mapping[self.power_level-1]:
            reward -= 1.5
        # small penalty for choosing to handoff when there's availibity
        if self.user_attached <= self.mapping[self.power_level] and dropped == 0 and action[1] == 1:
            reward -= 1
        # small bonus for choosing to handoff when not enough resources
        if self.user_attached == self.mapping[self.power_level] and dropped > 0 and action[1] == 1:
            reward += 1

        return reward
    