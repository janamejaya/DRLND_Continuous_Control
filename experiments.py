import numpy as np
import random
import time
from collections import deque
import torch

class Continuous_Action_Experiments():
    """Interacts with and learns from the environment using discrete actions."""
    
    def __init__(self, nagents, nruns, nepisodes, maxt, start_eps, end_eps, decay_eps, current_agent, 
                 current_env, target_score, num_episodes_score_avg):
        """Initialize a Discrete Experiment object.
        Params
        ======
            nagents (int): Number of non-interacting agents
            nruns (int): Number of times the experiment will be run
            nepisodes (int): Number of episodes in each run
            maxt (int): Maximum number of steps per episode
            start_eps (float): starting value of epsilon, for epsilon-greedy action selection
            end_eps (float): minimum value of epsilon
            decay_eps (float): multiplicative factor (per episode) for decreasing epsilon
            current_agent (Class): selected agent
            current_env (Class): selected_environment
            target_score (float): target score to be achieved for successful run
            num_episodes_score_avg (int): number of scores over which a running average will be monitored
        """
        self.num_agents = nagents
        self.nruns = nruns
        self.nepisodes = nepisodes
        self.maxt = maxt
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_eps = decay_eps
        self.agent = current_agent
        self.env = current_env,
        self.target_score = target_score
        self.num_episodes_score_avg = num_episodes_score_avg
        #print('brain_name = ', self.env[0].brain_names[0])
        self.brain_name = self.env[0].brain_names[0]
        print('self.brain_name = ', self.brain_name)
        print('Continuous Action Experiment initialized')

    def execute_one_episode(self, current_eps):
        env_info = self.env[0].reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations   # get the next state
                    
        # Initialize the score for each agent
        scores = np.zeros(self.num_agents)
        
        # Reset the agent
        self.agent.reset()
        
        # For each time-step during the episode
        for t in range(self.maxt):
            # Select an action. Note these are continuous values
            actions = self.agent.act(states, self.agent.add_noise)
            
            # Take the action and collect the next state, reward and terminal state status
            env_info = self.env[0].step(actions)[self.brain_name]
            
            # Get the next state, reward, and episode termination info
            next_states = env_info.vector_observations        # get the next state
            rewards = env_info.rewards                        # get the reward
            dones = env_info.local_done                       # see if episode has finished
            #print('In execute one step, next_states.shape = ',next_states.shape, ' rewards.shape = ', rewards.shape,' dones.shape = ',dones.shape)
            
            # Here there are multiple agents, so for each agent update the agents over the transition from state to next_state
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.agent.step(state, action, reward, next_state, done, t)
            
            # Update state information for each agent
            states = next_states
            
            # Add reward to the score
            scores += rewards
            
            # If any of the 20 agents terminates, the episode ends
            if np.any(done):
                break
        return scores
    
    def execute_one_run(self, runid):
        """
        Execute one run with several episodes until the average score over the last num_episodes_score_avg
        exceeds the target value of target_score
        """
        #print('Inside execute run : ',runid)
        # Initialize scores
        scores = []
        scores_window = deque(maxlen=self.num_episodes_score_avg)
        
        # Initialize eps
        eps = self.start_eps
        
        # Reset the Actor and Critic networks
        self.agent.critic_local.reset_parameters()
        self.agent.actor_local.reset_parameters()
        self.agent.critic_target.reset_parameters()
        self.agent.actor_target.reset_parameters()
                            
        # Clear the experience buffer
        self.agent.memory.clear_buffer()
        
        # Start the timer
        start_time=time.time()
        print('\n')
        for episode_num in range(self.nepisodes): 
            # Run one episode and return the score for that episode
            # score corresponds to the total reward
            current_scores = self.execute_one_episode(eps)
            #print('episode_num = ', episode_num, ' current score = ',current_score)
            
            # Find the average score
            mean_score_per_agent = np.mean(current_scores)

            # Append current_score to the scores list
            scores.append(mean_score_per_agent)
            scores_window.append(mean_score_per_agent)
            
            #decrease epsilon
            eps = max(self.end_eps, eps*self.decay_eps)
            
            # Get the average score from scores_window
            avg_score = np.mean(scores_window)
            print('episode : ',episode_num, ' mean_score = ',mean_score_per_agent, ' avg score = ', avg_score)
            
            # Show the average score every num_episodes_score_avg time steps
            if episode_num%self.num_episodes_score_avg==0:
                print('\nRun {:d} \tEpisode_Num {:d} \tAverage Score: {:.2f}'.format(runid, episode_num, avg_score))

            # If avg_score exceeds the target score
            if avg_score>=self.target_score:
                end_time = time.time()
                
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} \tTotal time/seconds: {:.2f}'.format(episode_num, avg_score, end_time-start_time))
                torch.save(self.agent.actor_local.state_dict(), 'checkpoint_actor_local'+str(runid)+'.pth')
                torch.save(self.agent.critic_local.state_dict(), 'checkpoint_critic_local'+str(runid)+'.pth')
                
                break
        print('end of run : ',runid)
        end_time=time.time()
        
        # Return scores for this run, the number of episodes required and the total time taken
        return scores, avg_score, episode_num, end_time-start_time
    
    def run_experiment(self):
        print('Running experiment')
        # Initial list to store list of scores from each run
        all_scores = []
        all_num_episodes = []
        all_total_times = []
        all_avg_scores = []
        
        # For each run
        for current_runid in range(self.nruns):
            print('current experiment number = ', current_runid)

            # Perform the experiment for one run and return the scores, number of episodes and total time required
            scores, avg_score, num_episodes, total_time = self.execute_one_run(current_runid)
            
            # store the scores, num_episodes and total_time
            all_scores.append(scores)
            all_num_episodes.append(num_episodes)
            all_total_times.append(total_time)
            all_avg_scores.append(avg_score)
            
        # Find the average number of episodes required to reach the target score
        avg_number_of_episodes = np.mean(all_num_episodes)
        std_number_of_episodes = np.std(all_num_episodes)
        print('\nAverage number of episodes required to reach target score : {:2f} +/- {:2f}'.format(avg_number_of_episodes, std_number_of_episodes))
        
        # Find the average number time required to reach the target score
        #print('list of time taken = ', all_total_times)
        avg_time = np.mean(all_total_times)
        std_time = np.std(all_total_times)
        print('Average time/seconds per run required to reach target score : {:2f} +/- {:2f}'.format(avg_time, std_time))
        
        # Return all scores
        return all_scores, all_avg_scores
    
