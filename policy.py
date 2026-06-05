import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):
    """
    RL policy and update rules
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output_dim = output_dim
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    def get_actions_batch(self, states, deterministic=False, exploration=0.01):
        dev = next(self.parameters()).device
        states_t = torch.tensor(states, dtype=torch.float32).to(dev)
        probs, values = self.forward(states_t)
        values = values.squeeze(-1)

        probs_np = probs.detach().cpu().numpy()
        N = len(states)
        actions = np.zeros(N, dtype=np.int64)

        for i in range(N):
            if deterministic:
                actions[i] = np.argmax(probs_np[i])
            elif random.random() < exploration:
                actions[i] = random.randint(0, self.output_dim - 1)
            else:
                actions[i] = np.random.choice(self.output_dim, p=probs_np[i])

        actions_t = torch.tensor(actions, dtype=torch.int64).to(dev)
        log_probs = torch.log(probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-9)

        return actions, log_probs, values

    @staticmethod
    def compute_gae(rewards, values, masks, bootstrap_value, gamma=0.999, lam=0.95):
        T, N = rewards.shape
        advantages = torch.zeros(T, N, device=rewards.device)
        gae = torch.zeros(N, device=rewards.device)

        next_value = bootstrap_value
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    @staticmethod
    def update_ppo(network, states, actions, old_log_probs, returns, advantages,
                   ppo_epochs=4, mini_batch_size=512, clip_eps=0.2,
                   vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        T_N = states.shape[0]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(ppo_epochs):
            indices = np.random.permutation(T_N)
            for start in range(0, T_N, mini_batch_size):
                batch_idx = indices[start:start + mini_batch_size]

                mb_states = states[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_returns = returns[batch_idx]
                mb_advantages = advantages[batch_idx]

                probs, values = network.forward(mb_states)
                values = values.squeeze(-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = vf_coef * (mb_returns - values).pow(2).mean()
                entropy_loss = -ent_coef * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                network.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                network.optimizer.step()

    @staticmethod
    def update_ac_parallel(network, rewards, log_probs, values, masks, bootstrap_values, gamma=0.99):
        T, N = rewards.shape

        returns = torch.zeros(T, N, device=rewards.device)
        R = bootstrap_values.detach()
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R * masks[t]
            returns[t] = R

        returns_flat = returns.view(-1).detach()
        log_probs_flat = log_probs.view(-1)
        values_flat = values.view(-1)

        advantage = returns_flat - values_flat
        actor_loss = (-log_probs_flat * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

