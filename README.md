## Implementation of Reinforcement Learning Algorithm

- Classification of Reinforcement Learning Algorithm
- ![image-20210312120614455](/home/czk119/Desktop/RL-learning/map.png)

#### Reinforce Algorithm

$$
\theta = argmax \ U(\theta) = argmax \ \sum_{\tau}P(\tau;\theta)*R(\tau)\\
P(\tau;\theta) = probability\ \ of \ \ trajectory \ \ \tau \ \ under \ \ \theta , \ \ U(\theta)  \ \ refers \ \ to \ \ the\ \ policy\\
R(\tau) = \sum_t \ R(s_t,a_t) \\
\frac{\partial U(\theta)}{\partial \theta}= \sum_{\tau} P(\tau;\theta)*\frac{\partial P(\tau;\theta)}{\partial \theta * P(\tau;\theta)}*R(\tau)=\sum_{\tau}\frac{\partial \log P(\tau;\theta)}{\partial \theta}*P(\tau;\theta)*R(\tau) =E(\frac{\partial \log P(\tau;\theta)}{\partial \theta}*R(\tau))\\
$$

