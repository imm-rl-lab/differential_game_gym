# Differential Game Gym

The repository contains examples of finite-horizon zero-sum differential games implemented as environments (Markov games) for multi-agent reinforcement learning algorithms. Since the problems are initially described by differential equations, in order to formalize them as Markov games, a uniform time-discretization with the diameter <code>dt</code> is used. In addition, it is important to emphasize that, in the games with a finite horizon, agent's optimal policies depend not only on the phase vector $x$, but also on the time $t$. Thus, we obtain Markov games, depending on <code>dt</code>, with continuous state space $S$ containing states $s=(t,x)$ and continuous action space $A$.

## Interface

The finite-horizon zero-sum differential games are implemented as environments (Markov games) with an interface close to [OpenAI Gym](https://www.gymlibrary.ml/) with the following attributes: 

- <code>state_dim</code> - the state space dimension; 
- <code>u_action_dim</code> - the action space dimension of the first agent;
- <code>v_action_dim</code> - the action space dimension of the second agent;
- <code>terminal_time</code> - the action space dimension;
- <code>dt</code> - the time-discretization diameter;
- <code>reset()</code> - to get an initial <code>state</code> (deterministic);
- <code>step(u_action, v_action)</code> - to get <code>next_state</code>, current <code>reward</code>, <code>done</code> (<code>True</code> if <code>t > terminal_time</code>, otherwise <code>False</code>), <code>info</code>;
- <code>virtual_step(state, u_action,v_action)</code> - to get the same as from <code>step(action)</code>, but but the current <code>state</code> is also set.
