## Implementation of Reinforcement Learning Algorithm

This is my python library and notes for Reinforcement Learning. Hope I can understand these algorithms completely.

### Key Concepts

- Bellman Equations
  $$
  \begin{equation}
  	\begin{split}
  	& V^{\pi}(s)= \mathop{E}_{a\sim\pi}[r(s,a)+\gamma V^\pi(s')] \\
  	& Q^{\pi}(s,a)=\mathop{E}_{s'\sim P}[r(s,a)+\gamma \mathop{E}_{a'\sim\pi}[Q^\pi(s',a')]] \\
  	& V^\pi(s)=\mathop{E}_{a\sim \pi}[Q^\pi(s,a)]
  	\end{split}
  \end{equation}
  $$
  Q(s,a) is Action-value Function and V(s) is value Function

- Advantage Functions
  $$
  \begin{equation}
  A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
  \end{equation}
  $$

### Kinds of Algorithm

![rl_algorithms_9_15](./image/rl_algorithms_9_15.svg)

- Q-Learning

  $\begin{equation}
  	\begin{split}
  		& Q^{new}(s_t,a_t)=(1-\alpha)Q(s_t,a_t)+\alpha(r_t+\gamma \max_\alpha(Q(s_{t+1},a))) \\
  		& \alpha \sim Learning \ \ rate，\gamma \sim Discount \ \ factor
  	\end{split}
  \end{equation}$

- Double Q-Learning
  $$
  Q_{truth}=
  $$
  

$$
\begin{equation}
	\begin{split}

	\end{split}
\end{equation}
$$

