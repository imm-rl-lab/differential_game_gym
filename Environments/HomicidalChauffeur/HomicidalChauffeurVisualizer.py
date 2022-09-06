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
        v_actions = np.array([np.mean([session['v_actions'][j] for session in [sessions[-1]]]) 
                              for j in range(len(sessions[0]['v_actions']))])

        plt.figure(figsize=[18, 12])
        plt.subplot(231)
        plt.plot(states[-1][1],states[-1][2],'bo', label='u_agent')
        plt.plot(states[-1][4],states[-1][5],'bo', label='v_agent')
        plt.plot([state[1] for state in states],[state[2] for state in states],'m--', label='u_agent')
        plt.plot([state[4] for state in states],[state[5] for state in states],'m--', label='v_agent')
        plt.xlim((0,15))
        plt.ylim((-5,15))
        plt.legend(loc='upper right')
        plt.grid()

        plt.subplot(232)
        plt.plot(np.arange(len(u_actions)) * env.dt, [u_action for u_action in u_actions],'g', label='u_actions')
        plt.xlim((0, env.terminal_time))
        plt.ylim((env.u_action_min[0] * 1.1, env.u_action_max[0] * 1.1))
        plt.legend()
        plt.grid()

        plt.subplot(233)
        plt.plot(np.arange(len(v_actions)) * env.dt, [v_action for v_action in v_actions],'g', label='v_actions')
        plt.xlim((0, env.terminal_time))
        plt.ylim((env.v_action_min[0] * 1.1, env.v_action_max[0] * 1.1))
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