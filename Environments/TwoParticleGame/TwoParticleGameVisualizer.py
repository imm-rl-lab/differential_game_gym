import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class TwoParticleGameVisualizer:
    def __init__(self, waiting_for_show=10, span=100):
        self.waiting_for_show = waiting_for_show
        self.span = span
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        return None

    def clean(self):
        self.total_rewards = []
        self.u_noise_thresholds = []
        self.v_noise_thresholds = []
        return None

    def show_fig(self, env, u_agent, v_agent, sessions):
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0)
                           for j in range(len(sessions[0]['states']))])
        u_actions = np.array([np.mean([session['u_actions'][j] for session in sessions])
                              for j in range(len(sessions[0]['u_actions']))])
        v_actions = np.array([np.mean([session['v_actions'][j] for session in sessions])
                              for j in range(len(sessions[0]['v_actions']))])
        plt.figure(figsize=[18, 12])

        plt.subplot(231)
        plt.plot([state[0] for state in states], [state[3] for state in states], 'r--', label='v-agent trajectory')
        plt.plot([state[0] for state in states], [state[1] for state in states], 'b--', label='u-agent trajectory')
        plt.plot(states[-1][0], states[-1][3], 'ro', label='terminal state for v-agent')
        plt.plot(states[-1][0], states[-1][1], 'bo', label='terminal state for u-agent')
        plt.legend()
        plt.xlim((0, env.terminal_time + 1))
        plt.grid()

        plt.subplot(232)
        plt.plot(np.arange(len(u_actions)) * env.dt, [u_action for u_action in u_actions], 'g', label='u-agent actions')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(233)
        plt.plot(np.arange(len(v_actions)) * env.dt, [v_action for v_action in v_actions], 'g', label='v-agent actions')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(234)
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n min={min(self.total_rewards):.2f} \n max={max(self.total_rewards):.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(235)
        plt.plot(self.u_noise_thresholds, 'g', label='u-agent noise threshold')
        plt.legend()
        plt.grid()

        plt.subplot(236)
        plt.plot(self.v_noise_thresholds, 'k', label='v-agent noise threshold')
        plt.legend()
        plt.grid()

        clear_output(True)

        plt.show()

    def show(self, env, u_agent, v_agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        self.total_rewards.append(total_reward)

        if episode % self.waiting_for_show == 0:
            self.show_fig(env, u_agent, v_agent, sessions)
            
        return None
