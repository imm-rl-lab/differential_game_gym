import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class UnequalGameVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []

    def clean(self):
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        
    def show_fig(self, env, u_agent, v_agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        u_actions = np.array([np.mean([session['u_actions'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['u_actions']))])
        v_actions = np.array([np.mean([session['v_actions'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['v_actions']))])
        plt.figure(figsize=[18, 12])

        plt.subplot(231)
        plt.plot([state[0] for state in states], [state[1] for state in states], 'm--', label='Траектория движения')
        plt.plot(states[-1][0],states[-1][1],'bo', label='Финальное состояние')
        plt.xlim((0, env.terminal_time))
        plt.grid()
        
        plt.subplot(232)
        plt.step(np.arange(len(u_actions)) * env.dt, [u_action for u_action in u_actions], 'g', label='Реализация U')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(233)
        plt.step(np.arange(len(v_actions)) * env.dt, [v_action for v_action in v_actions], 'g', label='Реализация V')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(234)
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n min={min(self.total_rewards):.2f} \n max={max(self.total_rewards):.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(235)
        plt.plot(self.u_noise_thresholds, 'g', label='Порог шума u-агента')
        plt.legend()
        plt.grid()

        plt.subplot(236)
        plt.plot(self.v_noise_thresholds, 'k', label='Порог шума v-агента')
        plt.legend()
        plt.grid()

        clear_output(True)
        
        plt.show()

    def show(self, env, u_agent, v_agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.total_rewards.append(total_reward)
        self.u_noise_thresholds.append(u_agent.noise.threshold)
        self.v_noise_thresholds.append(v_agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, u_agent, v_agent, sessions)
