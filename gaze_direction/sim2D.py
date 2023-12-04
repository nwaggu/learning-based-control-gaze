import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy

class SIM2D:
    def __init__(self, x_locations=None, y_locations=None) -> None:

        # initialize grid world
        self.grid_world = (12, 10)  # bounds of world
        self.fov = [4,3]    # 'robot' (x,y) field of view
        self.gaze_pos = None    # where the 'robot' is looking
        self._initialize_gaze_centered()

        # action space
        # self.actions = ["up", "down", "right", "left", "no"]
        self.actions = [0, 1, 2, 3, 4]

        # initialize people locations and speakers
        self.speaker_fraction = 1/3
        self.num_people = None
        self.num_speakers = None
        self.people_loc = None
        self.moving_people = None
        self._initialize_people(x_locations, y_locations)
        self._initialize_moving_people()
        
    def _initialize_gaze_random(self):
        x = float(np.random.randint(0, self.grid_world[0]))
        y = float(np.random.randint(0, self.grid_world[1]))
        self.gaze_pos = np.array([x,y])

    def _initialize_gaze_centered(self):
        x = float(self.grid_world[0]/2)
        y = float(self.grid_world[1]/2)
        self.gaze_pos = np.array([x,y])
    
    def _initialize_people(self, x_locations=None, y_locations=None):
        """Randomizes the number of people present and the number of speakers
            Creates those people as points on a gridworld"""
        # self.num_people = np.random.randint(5, 20)
        if x_locations is not None:
            self.num_people = len(x_locations)
        else:
            self.num_people = 10

        self.num_speakers = int(self.num_people*self.speaker_fraction)
        self._initialize_locations(x_locations, y_locations)
        self._initialize_speakers()

    def _initialize_locations(self, x_locations=None, y_locations=None):
        """Creates self.num_people number of random points within the total field of view"""
        if x_locations is None:
            x_locations = np.random.uniform(0, self.grid_world[0]-1, size=(self.num_people, 1))
        if y_locations is None:
            y_locations = np.random.uniform(0, self.grid_world[1]-1, size=(self.num_people, 1))

        is_speaker = np.zeros((self.num_people, 1))
        self.people_loc = np.column_stack([x_locations, y_locations, is_speaker])
        
    def _initialize_speakers(self):
        """Grabs self.num_speakers number of random points from the people locations
            and assigns them as speakers"""
        if self.num_speakers > 0:
            # pick a self.num_speakers points from the people x-locations
            x_locs = np.random.choice(self.people_loc[:,0], size=(self.num_speakers,1), replace=False)
            # find the corresponding y-locations 
            speaker_indices = np.asarray([np.where(self.people_loc == x)[0][0] for x in x_locs])
            # for each speaker index, assign the relevant location in people_loc to a "1"
            self.people_loc[:,2][speaker_indices] = 1

    def _initialize_moving_people(self, max_moving=2):
        """Picks up to 3 people to move on a "step" and assigns them goal positions"""
        num_moving_people = np.random.randint(1,max_moving+1)
        moving_people_indices = np.random.randint(0, self.num_people, num_moving_people)
        xgoals = np.random.uniform(0, self.grid_world[0]-1, num_moving_people)
        ygoals = np.random.uniform(0, self.grid_world[1]-1, num_moving_people)
        self.moving_people = np.vstack((moving_people_indices, xgoals, ygoals))

    def sample_action_discrete(self):
        return np.random.choice(self.actions)
    
    def sample_action_continuous(self):
        """Currently returns (Δx, Δy) to move"""

        # ensures we don't leave the grid world; takes the smaller of the distances to either edge
        # as the maximum possible amount to move
        xmax = min(abs(self.grid_world[0]-self.gaze_pos[0]), self.gaze_pos[0])/10
        ymax = min(abs(self.grid_world[1]-self.gaze_pos[1]), self.gaze_pos[1])/10

        delx = np.random.uniform(-xmax, xmax)
        dely = np.random.uniform(-ymax, ymax)

        return delx, dely
    
    def return_state(self):
        """Returns a np array of all people locations in robot frame and their speaker tag"""

        people_robot_frame = np.round(self.people_loc[:,0:2] - self.gaze_pos, 2)

        state = np.ones((1, len(self.people_loc[0]), len(self.people_loc)))
        for i in range(len(people_robot_frame[0])):
            state[0][i] = people_robot_frame[:,i]

        state[0,2] = self.people_loc[:,2]

        return state

    def reset(self, moving_people=False):
        self._initialize_gaze_centered()
        # self._initialize_people()
        if moving_people:
            # self._initialize_moving_people()
            self._move_people()
        return deepcopy(self.return_state())

    def reset_discrete(self):
        self._initialize_gaze_random()
        return deepcopy(self.gaze_pos)

    def _is_valid(self, position):
        """Checks if the gaze position of the robot is within the grid world bounds"""
        x,y = position
        if x < 0 or x > (self.grid_world[0]-1):
            return False
        if y < 0 or y > (self.grid_world[1]-1):
            return False
        return True
    
    def _is_inside_fov(self, point, position):
        """Checks if one point is inside the robot's field of view"""
        x,y = point
        xmin = position[0] - self.fov[0]/2
        xmax = position[0] + self.fov[0]/2
        ymin = position[1] - self.fov[1]/2
        ymax = position[1] + self.fov[1]/2

        if x < xmin or x > xmax:
            return False
        if y < ymin or y > ymax:
            return False
        return True
    
    def num_inside_fov(self):
        num = 0
        for point in self.people_loc[:,0:2]:
            if self._is_inside_fov(point, self.gaze_pos):
                num += 1

        return num

    def _get_reward(self):
        """Returns the reward based on the number of people inside the robot's FOV"""
        reward = -1
        state = self.return_state()[0]
        xlocs, ylocs, speaking = state

        for i, x in enumerate(xlocs):
            y = ylocs[i]
            if -self.fov[0]/2 < x < self.fov[0]/2 and -self.fov[1]/2 < y < self.fov[1]/2:
                reward += 1
                if speaking[i] == 1:
                    reward += 2

        return reward

    def _take_action_discrete(self, action, position):
        x,y = position
        if action == 0:
            y+=1
        elif action == 1:
            y-=1
        elif action == 2:
            x+=1
        elif action == 3:
            x-=1
        elif action == 4:
            pass
        if self._is_valid([x,y]):
            return [x,y]
        else:
            return position
        
    def _move_people(self):
        """Moves the locations in self.people_loc according to the 
            locations in self.moving_people. Only moves by a fraction of the total"""
        for i, index in enumerate(self.moving_people[0,:]):
            index = int(index)
            current_person_loc = self.people_loc[index][0:2]
            goal = self.moving_people[1:3,i]
            move_by = goal - current_person_loc
            move_by /= 15

            self.people_loc[index][0:2] += move_by

    def step_discrete(self, action):
        self.gaze_pos = self._take_action_discrete(action, self.gaze_pos)
        reward = self._get_reward()
        return deepcopy(self.gaze_pos), reward
    
    def step_continuous(self, action, moving_people=False):
        """Moves the gaze by (Δx, Δy). Should eventually move to (θ, φ) angle pan
            Inputs: np.array([Δx, Δy])"""
        terminated = False

        # print(self.gaze_pos)
        # print(action[0])

        if moving_people:
            self._move_people()

        self.gaze_pos += action
        reward = self._get_reward()

        if not self._is_valid(self.gaze_pos):
            terminated = True
            reward -= 1000
        
        return deepcopy(self.return_state()), reward, terminated

    def make_plot(self, ax=None, title="Gridworld Representation", show=True):
        if ax is None:
            fig, ax = plt.subplots(1,1)

        # plot the people, speakers circled in red
        ax.scatter(self.people_loc[:,0], self.people_loc[:,1], label="People")
        speakers_x = []
        speakers_y = []
        for i, val in enumerate(self.people_loc[:,-1]):
            if val == 1:
                speakers_x.append(self.people_loc[i,0])
                speakers_y.append(self.people_loc[i,1])

        ax.scatter(speakers_x, speakers_y, s=80, facecolors='none', edgecolors='r', label="Speakers")
        # ax.scatter(self.speakers_loc[:,0], self.speakers_loc[:,1], s=80, facecolors='none', edgecolors='r')

        # plot the 'robot' field of view centerpoint and square
        ax.plot(self.gaze_pos[0], self.gaze_pos[1], 'y+', label="Gaze Center")
        rect_ll = (self.gaze_pos[0] - self.fov[0]/2, self.gaze_pos[1] - self.fov[1]/2)
        fov_rect = Rectangle(rect_ll, self.fov[0], self.fov[1], facecolor='none', edgecolor='y')
        ax.add_patch(fov_rect)

        ax.set_xlim(-1, self.grid_world[0])
        ax.set_ylim(-1, self.grid_world[1])

        ax.set_title(title)
        ax.legend()
        
        if show:
            plt.show()

        return ax
    
    def plot_state(self):
        fig, ax = plt.subplots(1,1)
        state = self.return_state()[0]
        ax.scatter(state[0], state[1])

        for i, val in enumerate(state[2]):
            if val == 1:
                ax.scatter(state[0][i], state[1][i], s=80, facecolors='none', edgecolors='r')

        plt.show()


if __name__ == "__main__":
    mysim = SIM2D()

    reward = mysim._get_reward()
    print(f"reward: {reward}")

    # point = mysim.gaze_pos - np.array([1,1])
    # print(mysim._is_inside_fov(point, mysim.gaze_pos))

    mysim.make_plot()
    # print(mysim.moving_people)
    # mysim.step_continuous([1,1], moving_people=True)
    # mysim.make_plot()

    # print(mysim.gaze_pos, mysim._get_reward())
    # mysim.make_plot()
    

    # mysim.make_plot()
    # mysim.plot_state()

    # move = np.array([2,4])
    # print(mysim.step_continuous(move))

    # mysim.make_plot()


    """
    Gameplan:
        state is represented in robot relative coordinate frame
        output representation remains the same, world doesn't change as robot moves
        need state representation with robot as (0,0)
        """

    
