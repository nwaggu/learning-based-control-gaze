import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sim2D import SIM2D

## ------------------------ global variables ------------------------ ##

# env = SIM2D()

# intialize q matrix to match grid world and action space dimensions, currently 10x12x5
def reset_Q_matrix(env):
    Q_matrix = np.zeros((env.grid_world[0], env.grid_world[1], len(env.actions)))
    return Q_matrix

## ------------------------ helper functions ------------------------ ##

def choose_action(Q_matrix, state, epsilon=0.1):
    """Returns integer and string of the chosen action.
    Choses the next action to take with some randomness.
    Takes a random action ε% of the time"""

    # ε% of the time, choose a random action (0, 1, 2, 3, 4)
    if np.random.rand() < epsilon:
        action = np.random.randint(0,5)
    # otherwise, choose the action that corresponds to the highest Q value for that state
    else:
        action = np.argmax(Q_matrix[int(state[0]), int(state[1]), :])
        
    return action

def update_Q_sarsa(Q_matrix, state, action, next_state, next_action, reward, alpha=0.5, gamma=0.8):
    current_q_index = state[0], state[1], action
    next_q_index = next_state[0], next_state[1], next_action

    current_q = Q_matrix[current_q_index]
    next_q = Q_matrix[next_q_index]

    Q_matrix[current_q_index] = current_q + alpha*(reward + gamma*next_q - current_q)
    return Q_matrix

def update_Q_qlearn(Q_matrix, state, action, next_state, reward, alpha=0.5, gamma=0.8):
    current_q_index = int(state[0]), int(state[1]), action
    current_q = Q_matrix[current_q_index]

    max_next_q = np.max(Q_matrix[int(next_state[0]), int(next_state[1]), :])

    Q_matrix[current_q_index] = current_q + alpha*(reward + gamma*max_next_q - current_q)
    return Q_matrix

def visualizer(env):
    fig, ax = env.make_plot(show=False)
    ax.grid(which='both')

    best_actions = np.argmax(Q_matrix, axis=2)

    # draw the best action in the grid square
    arrows = [[0,0.5], [0,-0.5], [0.5,0], [-0.5,0], [0,0]]
    for i, row in enumerate(best_actions):
        x_pos = i
        for j, element in enumerate(row):  
            # plt.text(x_pos, j+0.25, env.actions[element])
            y_pos = j
            if element == 4:
                ax.plot(x_pos, y_pos, 'k.', alpha=0.25)
            else:
                ax.arrow(x_pos, y_pos, arrows[element][0], arrows[element][1], head_width=0.15, length_includes_head=True, facecolor='k', alpha=0.25)

    plt.show()

def cumulative_mean(array):
    cumulative_sum = np.cumsum(array, axis=0)
    mean = cumulative_sum / np.arange(1,len(array)+1)
    return mean       

## ------------------------ learning algorithm ------------------------ ##
def do_learning(env, Q_matrix, epochs, alpha, gamma):
    epoch_reward = []
    for learning_epoch in tqdm(range(epochs)):
        # reset the environment to the original configuration every episode
        state = env.reset_discrete()       
        # choose an initial state-action pair                   
        action = choose_action(Q_matrix, state)

        time_step_reward = 0
        for time_step in range(50):
            # take the chosen action and return the associated reward and new state
            next_state, reward = env.step_discrete(action)
            # do that again
            next_action = choose_action(Q_matrix, next_state)
            # use the current state and next state to update the Q matrix
            Q_matrix = update_Q_qlearn(Q_matrix, state, action, next_state, reward, alpha, gamma)
            # update loop params
            state = next_state
            action = next_action
            
            time_step_reward+= reward

        epoch_reward.append(time_step_reward)

    return Q_matrix, epoch_reward

def test(Q_matrix, env, visualizer=True):
    state = env.reset_discrete()                        
    action = choose_action(Q_matrix, state)
    reward_over_time = []
    for time_step in range(10):
        state, reward = env.step_discrete(action)
        action = choose_action(Q_matrix, state, epsilon=0)

        reward_over_time.append(reward)
        if visualizer:
            plt.ion()
            plt.plot(reward_over_time)
            plt.title("Immediate Reward")
            plt.xlabel("Time Steps")
            plt.ylabel("Reward")
            plt.pause(0.001)
            plt.show()

    plt.ioff()
    env.make_plot()


## ------------------------ flight code ------------------------ ##

if __name__ == "__main__":
    alg = "qlearn"
    reward_list = []
    trials = 1
    epochs = 700
    alpha = 0.5
    gamma = 0.7

    save_mean = False

    for _ in range(trials):
        env = SIM2D()
        Q_matrix = reset_Q_matrix(env)
        Q_matrix_final, reward = do_learning(env, Q_matrix, epochs, alpha, gamma)
        reward_list.append(cumulative_mean(reward))

    reward_over_trials = np.mean(reward_list, axis=0)
    std_over_trials = np.std(reward_list, axis=0)
    error_in_mean = std_over_trials / np.sqrt(trials)
    plt.plot(reward_over_trials)
    plt.fill_between(np.arange(0, epochs), reward_over_trials+error_in_mean, reward_over_trials-error_in_mean, alpha=0.4)
    plt.show()

    if save_mean:
        np.save("q_learn_mean-10_trials.npy", reward_over_trials)
        np.save("q_learn_err-10_trials.npy", error_in_mean)

    test(Q_matrix_final, env)


