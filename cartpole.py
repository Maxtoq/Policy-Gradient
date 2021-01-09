import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import numpy as np

import gym


def compute_mov_avg(data: list, window=10):
    ids = [window * x for x in range(0, int(len(data) / window) + 1)]
    if ids[-1] < len(data) - 1:
        ids.append(len(data) - 1)
    mov_avg = []
    for i in range(1, len(ids)):
        mov_avg.append(sum(data[ids[i - 1]:ids[i]]) / len(data[ids[i - 1]:ids[i]]))
    return ids[1:], mov_avg

def display_metrics(data: list, legend: str, mov_avg=True):
    plt.figure(figsize=(15,5))
    # Plot data
    plt.plot(np.arange(len(data)), data, label=legend)
    # Plot moving average
    if mov_avg:
        ids, avg = compute_mov_avg(data, 30)
        plt.plot(ids, avg, label=legend + ' average')
    plt.legend()
    plt.show()


class ReinforceAgent(object):

    def __init__(self, env, lr=0.01, dr=0.95):
        # Cartpole environment
        self.cartpole_env = env

        # Model to learn policy: 
        #   - Inputs: 4 real values: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        #   - 3 fully connected layers
        #   - final softmax activation to get probability distribution
        #   - Outputs: Probability of taking the two actions: [Push cart to left, Push cart to right]
        self.policy = nn.Sequential(
            nn.Linear(in_features=4, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=2),
            nn.Softmax()
        )

        # Learning rate
        self.lr = lr
        # Discount rate
        self.dr = dr

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Saving returns
        self.complete_returns = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        distrib = Categorical(probs)
        action = distrib.sample()
        return action.item(), distrib.log_prob(action)

    def train(self, nb_episodes=3000):
        for i in range(nb_episodes):
            state = env.reset()
            log_probs = []
            rewards = []
            for t in range(500):
                env.render()
                # Choose action
                action, log_prob = self.choose_action(state)
                # Save log probability of chosen action
                log_probs.append(log_prob)                
                # Perform action and retrieve new_state and reward
                state, reward, done, info = env.step(action)
                # Save reward
                rewards.append(reward)
                # Check for end state
                if done:
                    print("Episode #{} finished after {} timesteps".format(i + 1, t + 1))
                    break
            
            # Save complete return
            self.complete_returns.append(sum(rewards))

            # Training on trajectory
            policy_loss = []
            # For each step
            for j in range(len(log_probs)):
                # Compute return
                G = 0.0
                for k in range(j, len(rewards)):
                    G += rewards[j] * self.dr ** k
                # Compute policy loss
                loss = -log_probs[j] * G
                policy_loss.append(loss)

            # Backpropagate
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
        
        display_metrics(self.complete_returns, "Return")


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = ReinforceAgent(env)

    agent.train()

    env.close()