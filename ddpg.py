import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import (Actor, Critic)
from noise import OUNoise
from replaybuffer import ReplayBuffer


class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,
                 hidden_layers_actor=[32,32],
                 hidden_layers_critic=[32, 32, 32],
                 buffer_size=int(1e5), batch_size=128, gamma=0.99, tau=1e-3,
                 learning_rate_actor=1e-4, learning_rate_critic=5e-4,
                 weight_decay=0.0001,
                 update_every=20, num_batches=10, add_noise=True, 
                 head_name_actor='Actor', head_name_critic="DuelingDQN",
                 head_scale_actor='max', head_scale_critic="max"):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hidden_layers (list of int ; optional): number of each layer nodes
            buffer_size (int ; optional): replay buffer size
            batch_size (int; optional): minibatch size
            gamma (float; optional): discount factor
            tau (float; optional): for soft update of target parameters
            learning_rate_X (float; optional): learning rate for X=actor or critic
            update_every (int; optional): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.update_every = update_every
        self.num_batches = num_batches
        self.weight_decay_critic = weight_decay
        self.add_noise=add_noise

        # detect GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        
        ### SET UP THE ACTOR NETWORK ###
        # Assign model parameters and assign device
        model_params_actor  = [state_size, action_size, seed, hidden_layers_actor, head_name_actor, head_scale_actor]
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(*model_params_actor).to(self.device)
        self.actor_target = Actor(*model_params_actor).to(self.device)
        
        # Set up optimizer for the Actor network
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)       
        
        
        ### SET UP THE CRITIC NETWORK ###
        model_params_critic = [state_size, action_size, seed, hidden_layers_critic, head_name_critic, head_scale_critic]

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(*model_params_critic).to(self.device)
        self.critic_target = Critic(*model_params_critic).to(self.device)
        
        # Set up optimizer for the Critic Network
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay_critic)

        
        # Noise process
        self.noise = OUNoise(action_size, self.seed)

        # Initialize Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size,
                                   self.batch_size, seed, self.device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, timestep):
        # Save experience in replay memory                
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size and timestep % self.update_every == 0:
            for i in range(self.num_batches):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        
        # Go to evaluation mode and get Q values for current state
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
            
        # get back to train mode
        self.actor_local.train()
        
        # Add noise to the action probabilities
        if add_noise:
            action_values += self.noise.sample()
        return np.clip(action_values, -1.0, 1.0)
    
    def reset(self):
        self.noise.reset()
   
    def learn(self, experiences, gamma):
        # From the experiences buffer, separate out S_t, A_t, R_t, S_t+1, done data
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks using the local and target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        X_target = tau*X_local + (1 - tau)*X_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)