#Реализация визуализатора
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PointOnThePlaneVisualizer:
    def __init__(self, waiting_for_show=10):
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        self.mean_reward = [0]
        self.waiting_for_show = waiting_for_show
    
    def show_fig(self, env, u_agent, v_agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        u_actions = np.array([np.mean([session['u_actions'][j] for session in sessions], axis=0) 
                              for j in range(len(sessions[0]['u_actions']))])
        v_actions = np.array([np.mean([session['v_actions'][j] for session in sessions], axis=0) 
                              for j in range(len(sessions[0]['v_actions']))])
        
        total_rewards = self.total_rewards[-1000:]
        mean_rewards = self.mean_reward[-1000:]
        plt.figure(figsize=[18, 9])
        
        plt.subplot(231)
        plt.plot(states[-1][1],states[-1][2],'bo', label='Terminal state')
        plt.plot([state[1] for state in states],[state[2] for state in states],'m--', label='Trajectory')
        plt.legend(loc='upper right')
        plt.grid()

        plt.subplot(432)
        plt.plot(np.arange(len(u_actions)) * env.dt, [u_action[0] for u_action in u_actions],'g', label='u_action[0]')
        plt.legend()
        plt.grid()
        
        plt.subplot(435)
        plt.plot(np.arange(len(u_actions)) * env.dt, [u_action[1] for u_action in u_actions],'g', label='u_action[1]')
        plt.legend()
        plt.grid()

        plt.subplot(433)
        plt.plot(np.arange(len(v_actions)) * env.dt, [v_action[0] for v_action in v_actions],'g', label='v_action[0]')
        plt.legend()
        plt.grid()
        
        plt.subplot(436)
        plt.plot(np.arange(len(v_actions)) * env.dt, [v_action[1] for v_action in v_actions],'g', label='v_action[1]')
        plt.legend()
        plt.grid()

        plt.subplot(234)
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n min={min(self.total_rewards):.2f} \n max={max(self.total_rewards):.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.plot(mean_rewards, 'b', label='mean_reward')
        plt.plot([0,len(total_rewards)], [0.8, 0.8], 'k', label='true value')
        plt.legend()
        plt.grid()
        
        plt.subplot(235)
        plt.plot(self.u_noise_thresholds, 'b', label='u-agent noise thresholds')
        plt.legend()
        plt.grid()
        
        plt.subplot(236)        
        plt.plot(self.v_noise_thresholds, 'b', label='v-agent noise thresholds')
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
            self.show_fig(env, u_agent, v_agent, sessions)
            
        return None
            
    def clean(self):
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        return None
