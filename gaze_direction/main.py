import numpy as np
import matplotlib.pyplot as plt

import qlearn
from neuroevolution import NeuroEvolution, NeuralNetwork
from sim2D import SIM2D

run_qlearn = False
run_neuroev = True

##  -------------- Globals  -------------- ##
env = SIM2D()
save_mean = False

## -------------- Run Qlearning  -------------- ##

if run_qlearn:
    reward_list = []
    trials = 5
    epochs = 700
    alpha = 0.5
    gamma = 0.7

    for _ in range(trials):
        env = SIM2D()
        Q_matrix = qlearn.reset_Q_matrix(env)
        Q_matrix_final, reward = qlearn.do_learning(env, Q_matrix, epochs, alpha, gamma)
        reward_list.append(qlearn.cumulative_mean(reward))

    reward_over_trials = np.mean(reward_list, axis=0)
    std_over_trials = np.std(reward_list, axis=0)
    error_in_mean = std_over_trials / np.sqrt(trials)
    plt.plot(reward_over_trials)
    plt.fill_between(np.arange(0, epochs), reward_over_trials+error_in_mean, reward_over_trials-error_in_mean, alpha=0.4)
    plt.show()

    if save_mean:
        np.save("q_learn_mean-10_trials.npy", reward_over_trials)
        np.save("q_learn_err-10_trials.npy", error_in_mean)

    # visualize the results of the last trial
    qlearn.test(Q_matrix_final, env)

## -------------- Run Neruoevolution -------------- ##

if run_neuroev:
    reward_list = []
    trials = 1
    epochs = 150
    num_networks = 5
    moving_people = False

    for _ in range(trials):
        alg = NeuroEvolution(num_networks)
        best_reward = alg.do_learning(epochs, moving_people)
        reward_list.append(best_reward)

    reward_over_trials = np.mean(reward_list, axis=0)
    std_over_trials = np.std(reward_list, axis=0)
    error_in_mean = std_over_trials / np.sqrt(trials)
    plt.plot(reward_over_trials)
    plt.fill_between(np.arange(0, epochs), reward_over_trials+error_in_mean, reward_over_trials-error_in_mean, alpha=0.4)
    plt.show()

    # visualize the results of the last trial
        # can use best_network to determine where to move given a state, or just take the final gaze pose
    best_network = alg._select_networks(1, test=True)[0]
    state_dict = best_network.state_dict()
    alg.test(state_dict, moving_people)

    if save_mean:
        np.save("q_learn_mean-10_trials.npy", reward_over_trials)
        np.save("q_learn_err-10_trials.npy", error_in_mean)

    best_direction_to_look = alg.env.gaze_pos
    dxdy = best_direction_to_look - alg.starting_gaze_position
    print(f"Gaze should move from {alg.starting_gaze_position} to {best_direction_to_look}, a change of {dxdy}")




