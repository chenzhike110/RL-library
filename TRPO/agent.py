import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import numpy as np
import scipy

from .model import critic, actor
from .utils import *

class agent(object):
    def __init__(self, num_inputs:int, num_outputs:int, opt) -> None:
        super().__init__()
        self.gamma = opt.gamma
        self.tau = opt.tau
        self.l2_reg = opt.l2_reg
        self.damping = opt.damping
        self.max_kl = opt.max_kl
        self.nsteps = opt.nsteps
        self.actor = actor(num_inputs, num_outputs)
        self.critic = critic(num_inputs)
    
    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.actor(state)
        action = torch.normal(action_mean, action_std)
        return action
    
    def update_critic(self, targets, states):
        """
        use scipy update critic parameters
        """
        def get_value_loss(flat_params):
            set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.critic(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.critic).data.double().numpy())

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.critic).double().numpy(), maxiter=25)
        set_flat_params_to(self.critic, torch.Tensor(flat_params))
    
    def update_actor(self, states, actions, advantages):
        action_means, action_log_stds, action_stds = self.actor(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
        
        loss = self.get_loss(states, actions, advantages, fixed_log_prob)
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def Fvp(v):
            kl = self.get_kl(states)
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * self.damping
        
        step_dir = conjugate_gradients(Fvp, -loss_grad, self.nsteps)
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / lm[0]

        neggdotstepdir = (-loss_grad * step_dir).sum(0, keepdim=True)
        print("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm())

        success, new_params = self.linesearch(states, actions, advantages, fixed_log_prob, fullstep, neggdotstepdir / lm[0])
        set_flat_params_to(self.actor, new_params)


    def update_params(self, batch):
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = critic(Variable(states))

        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        pre_return = 0
        pre_value = 0
        pre_advantage = 0

        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * pre_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * pre_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * pre_advantage * masks[i]

            pre_return = returns[i, 0]
            pre_value = values.data[i, 0]
            pre_advantage = advantages[i, 0]
        
        # update critic
        targets = Variable(returns)
        self.update_critic(targets, states)
        
        # update actor
        advantages = (advantages - advantages.mean()) / advantages.std()
        self.update_actor(states, actions, advantages)
    
    def linesearch(self, states, actions, advantages, fixed_log_prob, fullstep, expected_improve_rate, accept_ratio=.1, max_backtracks=10):
        fval = self.get_loss(states, actions, advantages, fixed_log_prob, True)
        print("fval before", fval.item())
        x = get_flat_params_from(self.actor)
        for _n_backtracks, stepfrac in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.actor, xnew)
            newfval = self.get_loss(states, actions, advantages, fixed_log_prob, True)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                print("fval after", newfval.item())
                return True, xnew
        return False, x

    def get_loss(self, states, actions, advantages, fixed_log_prob, volatile=False):
        if volatile:
            with torch.no_grad:
                action_means, action_log_stds, action_stds = self.actor(Variable(states))
        else:
            action_means, action_log_stds, action_stds = self.actor(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()
    
    def get_kl(self, states):
        mean1, log_std1, std1 = self.actor(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0 + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl
    
    def learn(self):
        pass
    

    

