import numpy as np


class UnequalGame:
    def __init__(self, dt=0.05, terminal_time=2, inner_step_n=10, initial_state=np.array([0, 1]),
                 u_action_min=np.array([-2]), u_action_max=np.array([2]),
                 v_action_min=np.array([-1]), v_action_max=np.array([1])):
        self.state_dim = 2
        self.u_action_dim = 1
        self.v_action_dim = 1
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.terminal_time = terminal_time
        self.initial_state = initial_state
        self.u_action_min = u_action_min
        self.u_action_max = u_action_max
        self.v_action_min = v_action_min
        self.v_action_max = v_action_max
        return None

    def reset(self):
        self.state = self.initial_state
        return self.state

    def dynamics(self, state, u_action, v_action):
        return np.array([1, u_action[0] + v_action[0]])
    
    def step(self, u_action, v_action):
        u_action = np.clip(u_action, self.u_action_min, self.u_action_max)
        v_action = np.clip(v_action, self.v_action_min, self.v_action_max)
        
        for _ in range(self.inner_step_n):
            K1 = self.dynamics(self.state, u_action, v_action)
            K2 = self.dynamics(self.state + K1 * self.inner_dt / 2, u_action, v_action)
            K3 = self.dynamics(self.state + K2 * self.inner_dt / 2, u_action, v_action)
            K4 = self.dynamics(self.state + K3 * self.inner_dt, u_action, v_action)
            self.state = self.state + (K1 + 2 * K2 + 2 * K3 + K4) * self.inner_dt / 6

        reward = 0
        done = False
        if self.state[0] >= self.terminal_time:
            reward = self.state[1] ** 2
            done = True
        
        return self.state, reward, done, None
