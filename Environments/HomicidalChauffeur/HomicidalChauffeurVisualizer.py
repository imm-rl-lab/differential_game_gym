import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class HomicidalChauffeurVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        return None
        
    def show_fig(self, env, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in [sessions[-1]]], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        u_actions = np.array([np.mean([session['u_actions'][j] for session in [sessions[-1]]]) 
                              for j in range(len(sessions[0]['u_actions']))])
        v_actions = np.array([np.mean([session['v_actions'][j] for session in [sessions[-1]]], axis=0) 
                              for j in range(len(sessions[0]['v_actions']))])

        plt.figure(figsize=[18, 12])
        plt.subplot(231)
        plt.plot(states[-1][1],states[-1][2],'bo', label='u_agent')
        plt.plot(states[-1][4],states[-1][5],'bo', label='v_agent')
        plt.plot([state[1] for state in states],[state[2] for state in states],'m--', label='u-agent trajectory')
        plt.plot([state[4] for state in states],[state[5] for state in states],'m--', label='v-agent trajectory')
        plt.xlim((-1,10))
        plt.ylim((-5,15))
        plt.legend(loc='upper right')
        plt.grid()

        plt.subplot(232)
        plt.plot(np.arange(len(u_actions)) * env.dt, [u_action for u_action in u_actions],'g', label='u-agent actions')
        plt.xlim((0, env.terminal_time))
        plt.ylim((env.u_action_min[0] * 1.1, env.u_action_max[0] * 1.1))
        plt.legend()
        plt.grid()

        plt.subplot(233)
        if env.v_action_dim == 2:
            _v_actions = self.scale_down_to_one(v_actions)
        else:
            _v_actions = np.array([[np.cos(v_action), np.sin(v_action)] for v_action in v_actions])
        plt.plot([v_action[0] for v_action in _v_actions], [v_action[1] for v_action in _v_actions],'g', label='v-agent actions')
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1))
        phi = np.linspace(0, 2 * np.pi, 50)
        plt.plot(np.sin(phi), np.cos(phi), alpha=0.5)
        plt.legend()
        plt.grid()

        plt.subplot(234)
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n min={min(self.total_rewards):.2f} \n max={max(self.total_rewards):.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(235)
        plt.plot(self.u_noise_thresholds,'g', label='u_noise_thrasholds')
        plt.legend()
        plt.grid()

        plt.subplot(236)
        plt.plot(self.v_noise_thresholds,'g', label='v_noise_thrasholds')
        plt.legend()
        plt.grid()
        
        clear_output(True)
        plt.show()
        
        return None
        
    def show(self, env, u_agent, v_agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.total_rewards.append(total_reward)
        self.u_noise_thresholds.append(u_agent.noise.threshold)
        self.v_noise_thresholds.append(v_agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, sessions)
            
        return None
            
    def clean(self):
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        return None
        
    def scale_down_to_one(self, v_actions):
        norm = np.linalg.norm(v_actions, axis=1)
        norm[norm < 1] = 1
        v_actions /= norm.reshape(norm.shape[0], 1)
        return v_actions