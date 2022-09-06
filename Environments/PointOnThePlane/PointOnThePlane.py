import numpy as np


class PointOnThePlane:
    def __init__(self, initial_state=np.array([0,0,2,2,1]), 
                 u_radius=np.array([6, 2]), v_radius=np.array([2, 4]),
                 terminal_time=4, dt=0.05, inner_step_n=10):
        self.state_dim = 5
        self.u_action_dim = 2
        self.v_action_dim = 2
        self.u_radius = u_radius
        self.v_radius = v_radius
        self.u_action_min = - np.ones(2)
        self.u_action_max = np.ones(2)
        self.v_action_min = - np.ones(2)
        self.v_action_max = np.ones(2)
        
        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        return None

    def state_coeffs(self, t):
        return np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [-4 * np.exp(t / 5), 0, -0.1 * np.exp(t / 5), 0],
                         [0, -4 * np.exp(t / 5), 0, -0.1 * np.exp(t / 5)]])

    def u_action_coeffs(self, t):
        return np.array([[ 0,  0],
                         [ 0,  0],
                         [-8,  0],
                         [ 0, -8]])

    def v_action_coeffs(self, t):
        return np.array([[0, 0],
                         [0, 0],
                         [2.4 * np.exp(t / 5), 0],
                         [0, 2.4 * np.exp(t / 5)]])

    def reset(self):
        self.state = self.initial_state
        return self.state
    
    def dynamics(self, state, u_action, v_action):
        t, x = state[0], state[1:]
        dx = np.dot(self.state_coeffs(t), x) + np.dot(self.u_action_coeffs(t), u_action) + np.dot(self.v_action_coeffs(t), v_action)
        return np.hstack((1, dx))

    def step(self, u_action, v_action):
        u_action = self.u_radius * u_action / max(np.linalg.norm(u_action), 1)
        v_action = self.v_radius * v_action / max(np.linalg.norm(v_action), 1)

        for _ in range(self.inner_step_n):
            K1 = self.dynamics(self.state, u_action, v_action)
            K2 = self.dynamics(self.state + K1 * self.inner_dt / 2, u_action, v_action)
            K3 = self.dynamics(self.state + K2 * self.inner_dt / 2, u_action, v_action)
            K4 = self.dynamics(self.state + K3 * self.inner_dt, u_action, v_action)
            self.state = self.state + (K1 + 2 * K2 + 2 * K3 + K4) * self.inner_dt / 6
        
        reward = 0
        done = False
        if self.state[0] >= self.terminal_time - self.dt / 2:
            reward = np.linalg.norm(self.state[1:3])
            done = True

        return self.state, reward, done, None
