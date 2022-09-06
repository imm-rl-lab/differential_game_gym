import numpy as np


class HomicidalChauffeur:
    def __init__(self, initial_state=np.array([0, 0, 0, 0, 2.5, 7.5]), dt=0.1, terminal_time=4, inner_step_n=10,
                 u_action_min=np.array([-1]), u_action_max=np.array([1]), 
                 v_action_min=np.array([0]), v_action_max=np.array([2 * np.pi])):
        self.state_dim = 6
        self.u_action_dim = 1
        self.v_action_dim = 1
        self.u_action_min = u_action_min
        self.u_action_max = u_action_max
        self.v_action_min = v_action_min
        self.v_action_max = v_action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
        return None

    def reset(self):
        self.state = self.initial_state
        return self.state

    def dynamics(self, state, u_action, v_action):
        return np.array([1, 
                        3 * np.cos(state[3]),
                        3 * np.sin(state[3]),
                        u_action[0],
                        np.cos(v_action[0]),
                        np.sin(v_action[0])
                        ])
    
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
        if self.state[0] >= self.terminal_time - self.dt / 2:
            reward = np.linalg.norm([self.state[1] - self.state[4], self.state[2] - self.state[5]])
            done = True

        return self.state, reward, done, None
