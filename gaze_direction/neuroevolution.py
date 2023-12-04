import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sim2D import SIM2D

## ------------- INITIALIZE PYTORCH ------------- ##
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

## ------------- CREATE NEURAL NETWORK ------------- ##

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.flatten = nn.Flatten()

        n_hidden = 10
        n_inputs = n_inputs
        n_outputs = 2

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

## ------------- CREATE NEUROEVOLUTION ALGORITHM ------------- ##

class NeuroEvolution(SIM2D):
    def __init__(self, num_networks) -> None:

        self.env = SIM2D()
        states = self.env.return_state()
        self.num_states = states.size
        self.network_pool = None
        self._initialize_networks(num_networks)
        self.env._initialize_gaze_centered()
        self.starting_gaze_position = self.env.gaze_pos

    def _generate_networks(self, num_networks):
        self.network_pool = np.zeros((num_networks, 2), dtype=object)
        for i, _ in enumerate(self.network_pool):
            self.network_pool[i,0] = NeuralNetwork(self.num_states).to(device)

    def _initialize_networks(self, num_networks):
        """Runs each network once to initialize their fitnesses"""
        self._generate_networks(num_networks)
        for i, val in enumerate(self.network_pool):
            # "val" is np.array of [network, fitness]
            network = val[0]
            state = self.env.reset()
            action = self.select_action(network, state)
            reward = self._run(state, action, network)
            self.network_pool[i,1] = reward

    def _rand_matrix_index(self, matrix):
        """ Mutation and crossover helper function.
            Returns (list) a random valid index within a matrix"""
        dimension = matrix.shape
        indices = []
        for dim in dimension:
            indices.append(np.random.randint(0, dim))

        return indices
    
    def _select_weight_matrix(self, model_dictionaries):
        """Mutation and crossover helper function.
            Takes in a list of dictionaries representing the model weights
            and biases (called by model.state_dict()), picks a random weight or bias
            matrix, and returns a list of each dict matrix"""
        # randomly select a weight or bias matrix
        model_state_dict = model_dictionaries[0]
        keys = list(model_state_dict.keys())
        # weight_matrix_key = np.random.choice(keys)
        weight_matrix_key = keys[0]
        # grab that matrix for all given inputs
        weight_matrices = [dict[weight_matrix_key] for dict in model_dictionaries]

        return weight_matrices, weight_matrix_key

    def mutate(self, model):
        """Mutates a network by randomly choosing one of four mutations:
            1. completely replacing with a new random value
            2. changing by some percentage (50% to 150%)
            3. adding a random number between -1 and 1
            4. changing the sign of the weight
            """
        # pull random weight or bias matrix from model
        state_dict_copy = copy.copy(model.state_dict())
        weight_matrix, key = self._select_weight_matrix([state_dict_copy])
        weight_matrix = weight_matrix[0]
        
        # pick a random value to change
        pos1, pos2 = self._rand_matrix_index(weight_matrix)

        prob = np.random.uniform()
        # replace with new random value
        if prob <= 0.25:
            new_rand_val = np.random.uniform()
            weight_matrix[pos1, pos2] = new_rand_val
        # change by some percentage
        elif prob > 0.25 and prob <= 0.5:
            scale = np.random.uniform(0.5, 1.5)
            weight_matrix[pos1, pos2] *= scale
        # add random number
        elif prob > 0.5 and prob <= 0.75:
            rand_addition = np.random.uniform(-1, 1)
            weight_matrix[pos1, pos2] += rand_addition
        # change sign
        else:
            weight_matrix[pos1, pos2] *= -1

        # load new state dictionary back into model
        state_dict_copy[key] = weight_matrix
        model.load_state_dict(state_dict_copy)
        return model
    
    def crossover(self, model1, model2):
        """Swaps either a weight (90% of the time) or a layer of weights (10% of the time) 
            between two networks to create a child
            Inputs: two parent neural networks
            Output: child neural network
            """
        # initialize a child
        child = copy.deepcopy(model1)
        # pull random weight or bias matrix for parents
        state_dict1 = copy.deepcopy(model1.state_dict())    # also the state dict for child
        state_dict2 = copy.deepcopy(model2.state_dict())
        weight_matrices, key = self._select_weight_matrix([state_dict1, state_dict2])
        weight_matrix1, weight_matrix2 = weight_matrices

        prob = np.random.uniform()
        # 10% of the time, swap a layer of weights (doesn't currently swap all bias terms)
        if prob < 0.1:
            layer = self._rand_matrix_index(weight_matrix1)[0]
            weight_matrix1[layer] = weight_matrix2[layer]
        # 90% of the time, swap one weight
        else:
            try:
                swap1, swap2 = self._rand_matrix_index(weight_matrix1)
                weight_matrix1[swap1, swap2] = weight_matrix2[swap1, swap2]
            # breaks for a 1D matrix (bias terms)
            except ValueError:
                swap = self._rand_matrix_index(weight_matrix1)[0]
                weight_matrix1[swap] = weight_matrix2[swap]
            
        state_dict1[key] = weight_matrix1 
        child.load_state_dict(state_dict1)
        return child

    def _select_networks(self, num_returned, epsilon=0.1, test=False):
        """Either selects the n best networks from the pool or choses n random networks from the pool.
        
        Returns a list of the selected n networks"""
        prob = np.random.uniform()
        # ε% of the time, choose a random network
        if prob < epsilon:
            networks = np.random.choice(self.network_pool[:,0], size=num_returned, replace=False)
        # otherwise, choose the network with the highest fitness (located in network_pool[:,1])
        else:
            # sort the fitnesses of each network
            sorted_network_indices = np.argsort(self.network_pool[:,1])
            # invert that so the highest fitness is at the top
            sorted_network_indices = sorted_network_indices[::-1]
            # pick the n best networks
            n_indices = sorted_network_indices[0:num_returned]
            best_n_networks = self.network_pool[n_indices]
            networks = best_n_networks[:,0]

        if test:
            return networks
        else:
            return copy.deepcopy(networks)
        
    def _remove_networks(self, num_removed):
        """Removes the n worst networks from the list"""
        # sort the fitnesses of each network
        sorted_network_indices = np.argsort(self.network_pool[:,1])
        # pick the n worst networks to remove
        n_indices = sorted_network_indices[0:num_removed]
        self.network_pool = np.delete(self.network_pool, n_indices, axis=0)

    def _insert_networks(self, network, fitness):
        """Adds new networks and corresponding fitness to self.network_pool.
            Inputs: lists of the networks and fitnesses to add"""
        for i, val in enumerate(network):
            self.network_pool = np.append(self.network_pool, [[val, fitness[i]]], axis=0)

    def select_action(self, network, state, epsilon=0.1):
        """Selects and action based on a network and state, with some randomness.
            in the gridworld, actions are currently (11/22) move by (Δx, Δy). Going 
            to eventualy move to (θ, φ) angle pan"""
        prob = np.random.uniform()
        # ε% of the time, move randomly
        if prob < epsilon:
            action = self.env.sample_action_continuous()
            action = torch.tensor([action], device=device, dtype=torch.float32, requires_grad=False)
        # otherwise, query the network
        else:
            state = torch.tensor(state, device=device, dtype=torch.float32, requires_grad=False)
            action = network(state)
            
        # print(gridworld.gaze_pos, action.tolist())
        action = np.array(action.tolist())
        action = np.round(action, 3)

        return action[0]
    
    def _run(self, state, action, network, max_steps=50, visualizer=False, epsilon=0.1, moving_people=False, track_gaze=False):
        """Runs a network for T time steps given an initial state and action. 
            Returns the total reward accumulated over T timesteps"""
        time_step_reward = 0
        reward_over_time = []
        gaze_positions = []
        ppl_locations = []
        gaze_positions.append(self.env.gaze_pos.tolist())
        reward_over_time.append(self.env._get_reward())
        ppl_locations.append(self.env.people_loc)

        for time_step in range(max_steps):
            state, reward, terminated = self.env.step_continuous(action, moving_people)
            action = self.select_action(network, state, epsilon=epsilon)
            if track_gaze:
                gaze_positions.append(np.round(self.env.gaze_pos,3).tolist())
                ppl_locations.append(self.env.people_loc)
            time_step_reward += reward
            reward_over_time.append(reward)

            if terminated:
                # print(f"terminated after {time_step} steps")
                break

            elif visualizer:
                plt.ion()
                plt.figure(1)
                
                plt.plot(reward_over_time)
                plt.title("Immediate Reward")
                plt.xlabel("Time Steps")
                plt.ylabel("Reward")

                plt.pause(0.001)
                plt.show()

        if visualizer:
            plt.ioff()
            self.env.make_plot(title="Final Gaze Position")

        if track_gaze:
            return time_step_reward, reward_over_time, gaze_positions, ppl_locations
        
        return time_step_reward
    
    def do_learning(self, epochs, moving_people=False):
        """Runs a genetic algorithm to modify the NN weights"""
        best_epoch_reward = []
        for learning_epoch in tqdm(range(epochs)):
            # pick network using epsilon-greedy alg
            network1, network2 = self._select_networks(2)
            # create child out of networks
            child = self.crossover(network1, network2)
            # randomly modify network parameters
            network1 = self.mutate(network1)
            network2 = self.mutate(network2)
            child = self.mutate(child)

            # use network on agent for T steps
            for network in [network1, network2, child]:
                state = self.env.reset(moving_people=moving_people)
                action = self.select_action(network, state)
                # evaluate network performance
                reward = self._run(state, action, network, moving_people=moving_people)
                # reinsert into pool
                self._insert_networks([network], [reward])
                # remove worst network from the pool
                self._remove_networks(1)

            best_epoch_reward.append(max(self.network_pool[:,1]))
        return best_epoch_reward
    
    def test(self, state_dict=None, moving_people=False, track_gaze=False, max_steps=20):
        """Loads the best state dictionary and runs it on a reset environment"""
        network = NeuralNetwork(self.num_states).to(device)

        if state_dict is None:
            state_dict = np.load("best_state_dict.npy", allow_pickle=True)[0]
        
        network.load_state_dict(state_dict)
        
        state = self.env.reset(moving_people)
        action = self.select_action(network, state, epsilon=0)

        if track_gaze:
            test_reward, reward_list, gaze_list, ppl_list = self._run(state, action, network, max_steps=max_steps, visualizer=True, epsilon=0, moving_people=moving_people, track_gaze=track_gaze)
            return test_reward, reward_list, gaze_list, ppl_list
        else:
            test_reward = self._run(state, action, network, max_steps=max_steps, visualizer=True, epsilon=0, moving_people=moving_people)
            return test_reward

## ------------- FLIGHT CODE ------------- ##
if __name__ == "__main__":
    epochs = 150
    num_networks = 5
    moving_people = False
    trials = 1
    save_results = False
    plot_results = False

    reward_list = []
    for _ in range(trials):
        alg = NeuroEvolution(num_networks)
        best_reward = alg.do_learning(epochs, moving_people)
        reward_list.append(best_reward)

    reward_over_trials = np.mean(reward_list, axis=0)
    std_over_trials = np.std(reward_list, axis=0)
    error_in_mean = std_over_trials / np.sqrt(trials)

    if save_results:
        np.save("ne_learn_mean-10_trials.npy", reward_over_trials)
        np.save("ne_learn_err-10_trials.npy", error_in_mean)

    if plot_results:
        fig, ax = plt.subplots(1,1)
        ax.plot(reward_over_trials)
        ax.fill_between(np.arange(0, epochs), reward_over_trials+error_in_mean, reward_over_trials-error_in_mean, alpha=0.4)
        ax.set_xlabel("epochs")
        ax.set_ylabel("reward")
        ax.set_title("Grid World Training Reward")
        plt.show()

        alg.env.make_plot()

    # plt.figure()
    best_network = alg._select_networks(1, test=True)[0]
    state_dict = best_network.state_dict()
    reward, reward_list, gaze_list, people_list = alg.test(state_dict, moving_people, track_gaze=True, max_steps=50)

    # save = input("Save weights? (y/n) \n")
    # if save == "y":
    #     np.save("best_state_dict.npy", np.array([state_dict]))

    fig, ax = plt.subplots(1,2)
    time_steps = np.arange(0, len(reward_list))
    def animate(i):
        ax[0].clear()
        ax[1].clear()
        alg.env.people_loc = people_list[i]
        alg.env.gaze_pos = gaze_list[i]
        alg.env.make_plot(ax[1])
        ax[0].plot(time_steps[0:i], reward_list[0:i])
        ax[0].set_ylim([-1, 8])

    # run the animation
    ani = FuncAnimation(fig, animate, len(gaze_list), repeat=False)
    plt.show()
    ani.save('test.gif')

        


