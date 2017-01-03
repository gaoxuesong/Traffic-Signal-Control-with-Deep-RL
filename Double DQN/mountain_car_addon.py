class addon(object):
    def __init__(self, adjustment_value):
        self.prev_action = -1
        self.adjustment_value = adjustment_value
    
    def reset(self):
        self.prev_action = -1
        
    def reward_adjusment(self, action):
        adjusted_reward = 0
        if (self.prev_action != -1) and (self.prev_action != action):
            adjusted_reward = self.adjustment_value
        self.prev_action = action
        
        return adjusted_reward
            