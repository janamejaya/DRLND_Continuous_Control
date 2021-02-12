## Report

The objective of the project was to train 20 independent agents each trying to move their double jointed arm in order to maintain contact with a target location.

### State and action space

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Reward

A reward of +0.1 was given each step for which the agents arm was at the correct location.

- Discount factor: 0.99

### Learning Algorithm

The approach for training the agent was based on the following factors

1. Each agent acts independently of the others, so only one agents policy needs to be trained
2. The agents controlled its actions by changing the torque applied to its two joints. The agents actions are continuous and the algorithm must be able to deal with them.
3. The experience of each of the 20 agents can be stored and combined in order to train the agent.
4. The algorithm should be fast and stable enough

Due to its similarity with [Deep Q-learning](https://www.nature.com/articles/nature14236) and ease of implementation, I selected the Deep Deterministic Policy Gradient (DDPG) algorithm.
The algorithm is summarized below

![](https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg)

### Function approximation with Neural Networks.

DDPG is an actor-critic algorithm for continuous actions. 

#### Actor

The Actor is the policy function approximator neural network which takes the current state as input and outputs the action an agent should take. The Actor neural network architecture can be simply described as follows

[State] -> [Hidden layer/384 neurons] -> [Hidden layer/256 neurons] -> Action [output/4-dimensional]

Since the actions are constrained to be in the range [-1,1], the tanh activation function was employed to generate the output. A rectified-linear activation function was used for all other hidden layers

As done for DQN, a local and a target network are used for the Actor. 

#### Critic

The Critic is the action-value function approximator neural network that takes the current state and selected action as input and outputs the action value for the given (state,action) pair. The Critic neural network architecture is shown next

[State,Action] -> [Hidden layer/384 neurons] -> [Hidden layer/256 neurons] -> Action-value function Q(s,a) [output/1-dimension]

Since Q(s,a) is a continuous number, a recified linear activation function was used for all hidden and output layers

A local and a target network were used for the Critic. 

#### Neural Network hyperparameters for both Actor and Critic
- Optimizer: Adam
- Learning rate for Actor: 0.0001
- Learning rate for Critic: 0.0005
- Update target network every : 20 steps

### Replay buffer and learning

- Buffer size: 100,000
- batch size for drawn experience = 128
- number of batches used for training = 10

#### Implementation
The algorithm implementation I used is based on the version provided by Udacity in the Deep Reinforcement Learning Nano-degree program github repository [https://github.com/udacity/deep-reinforcement-learning]


### Exprimental setup

Since neural network optimization and expoloration of the environment are stochastic processes, the experiment should be repeated several times. Accordingly, the following setup was used

- Maximum number of episodes per experiment: 2000
- Maximum number of steps per episode : 1000
- Number of experiments: 2

Ideally I would have used 10 independent experiments to evaluate the performance of the trained agent. However, due to gpu constraints, I performed two independent experiments.
Note that for each independent experiment, the neural network weights were initialized and the experience replay buffer were emptied.


### Criteria for success

When the running average of the per agent per episode reward over the last 100 episodes (if experiment lasts for over 100 episodes) consecutive episodes exceeds 30, the environment is deemed to be solved.

### Results
The number of episodes and total time required to successfully complete each round of the experiment was noted in order to quantify the success.

 The following average metrics were calculated when the experiments successfully completed.

1. Number of episodes required for completion - 65.500000 +/- 39.500000
2. Time (in seconds) required for completion - 8071.121342 +/- 4836.099544

Note that these numbers are not very reliable as they are based on only two experiments.

The performance (as quantified by the score) of the agent for each of these 2 experiments is displayed in the following ![figure](https://github.com/janamejaya/DRLND_Continuous_Control/blob/master/result_score.jpg)

### Future improvements

1. Improving the sampling: Although multiple independent agents did provide more diversity of experiences, perhaps multiple trajectories sampled asynchronously might help
2. Improving the algorithm: Perhaps incorporating double Q-learning, decoupling updates of the actor and critic networks by using different update frequencies, and smoothing the policy over states could help. It would be helpful to understand which aspect of different algorithms help improve the agents performance by getting benchmark results for the Reacher environment.
