"""Welcome to Spidy world!!!
Here our friendly neighbourhood sipder man saves the world from evil...
But will he be able to save his beloved Gwen Stacy??
Lets find out!!!

"""

# Importing all necessary libraries
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

class Spidy(gym.Env):                                                                   # Creating a gym env- Spidy
    def __init__(self, grid_size=6):                                                    
        super().__init__()                                                              # Inheriting all functions from gym.env
        self.grid_size = grid_size                                                      # The grid size
        self.agent_state = np.array([0, 0])                                             # Agent state, hell states and goal state
        self.goal_state = np.array([4, 5])  
        self.stacy_health = 100                                                         # Gwen Stacy's health
        self.timer = 60                                                                 # Countdown timer in seconds
        self.power_up_state = np.array([2, 4])                                          # This adds health if Spidy enter this state
        self.obstacle_states = [np.array([2, 2]), 
                                np.array([1, 4]), 
                                np.array([4, 2]),
                                np.array([3, 5]),
                                np.array([5, 4])]
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-0.5, high=self.grid_size-0.5, 
                                                shape=(2,), dtype=np.int32)
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

        self.background_img = mpimg.imread("Web.jpg")
        self.goal_img = mpimg.imread("Gwen Stacy.png")
        self.agent_img = mpimg.imread("Spiderman.png")
        self.power_up_img = mpimg.imread("Spiderman- Powerup.png")
        self.power_up_active = False
        self.obstacle_images = [mpimg.imread("Lizard.png"), 
                                mpimg.imread("Mysterio.png"), 
                                mpimg.imread("Electro.png"),
                                mpimg.imread("Green Goblin.png"),
                                mpimg.imread("Doctor Octopus.png")]
        
        """Once again Spidy faces a dilemma whether to save people or Gwen
        We all know what happened last time... but life gives everyone another chance
        Will Spidy be able to save Gwen this time around by evading Lizard, Electro and Mysterio...?
        
        """   

    def reset(self):                                                                    # Reset method resets the agent to starting point
        self.agent_state = np.array([0, 0])
        self.start_time = time.time()   
        self.stacy_health = 100
        self.stacy_health = 100                                                         # Reset health
        self.timer = 60                                                                 # Reset timer to 60 seconds
        return self.agent_state

    def step(self, action):                                                             # Step funcion decides the agents next action
        if action == 0 and self.agent_state[1] < self.grid_size - 1:                    # to move up, used -1 so that whole agent image is visible
            self.agent_state[1] += 1 # Right
        elif action == 1 and self.agent_state[1] > 0:                                   # to move down
            self.agent_state[1] -= 1 # Left
        elif action == 2 and self.agent_state[0] > 0:                                   # to move left
            self.agent_state[0] -= 1 # Up
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:                  # to move right, used -1 so that whole agent image is visible
            self.agent_state[0] += 1 # Down

        done = False                                                                    # Assigning rewards to different states
        if np.array_equal(self.agent_state, self.power_up_state):
            self.stacy_health = min(self.stacy_health + 20, 100)
            reward = 2
        elif np.array_equal(self.agent_state, self.goal_state):
            reward = 15
            done = True
        elif any(np.array_equal(self.agent_state, obs) for obs in self.obstacle_states):
            reward = -10
            done = True
        elif self.timer <= 0 or self.stacy_health <= 0:
            reward = -10
            done = True
        else:
            reward = -1

        self.stacy_health -= 1                                                          # Decreasing health and counter over time
        elapsed_time = time.time() - self.start_time
        self.timer = 60 - int(elapsed_time)

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)

        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):                                                                   # Redering all characters and background
        self.ax.clear()
        self.ax.imshow(self.background_img, 
                       extent=[-1, self.grid_size, -1, self.grid_size])                 

        goal_extent = [self.goal_state[0] - 0.5, self.goal_state[0] + 0.5, 
                       self.goal_state[1] - 0.5, self.goal_state[1] + 0.5]
        self.ax.imshow(self.goal_img, extent=goal_extent)

        agent_extent = [self.agent_state[0] - 0.5, self.agent_state[0] + 0.5, 
                        self.agent_state[1] - 0.5, self.agent_state[1] + 0.5]
        self.ax.imshow(self.agent_img, extent=agent_extent)

        power_up_extent = [self.power_up_state[0] - 0.5, self.power_up_state[0] + 0.5, 
                           self.power_up_state[1] - 0.5, self.power_up_state[1] + 0.5]
        self.ax.imshow(self.power_up_img, extent=power_up_extent)

        for state, img in zip(self.obstacle_states, self.obstacle_images):
            obstacle_extent = [state[0] - 0.5, state[0] + 0.5, 
                               state[1] - 0.5, state[1] + 0.5]
            self.ax.imshow(img, extent=obstacle_extent)

        self.ax.text(0.5, self.grid_size + 0.5, f'Timer: {self.timer}', ha='center')
        self.ax.text(5, self.grid_size + 0.5, f'Stacy Health: {self.stacy_health}', ha='center')

        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.01)

    def close(self):                                                                    # Close the env
        plt.close()

# if __name__ == "__main__":
#     env = Spidy()
#     state = env.reset()
#     for _ in range(500):
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         env.render()
#         print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
#         if done:
#             if np.array_equal(state, env.goal_state):
#                 print("Spidy Saves Stacy")
#                 env.ax.clear()
#                 env.final_image = mpimg.imread("Final.jpg")
#                 env.ax.imshow(env.final_image, extent=[0, env.grid_size, 0, env.grid_size])
#                 plt.draw()
#                 plt.axis("off")
#                 plt.pause(3)
#             elif env.timer <= 0 or env.stacy_health <= 0:
#                 print("Stacy Dies again")
#                 env.ax.clear()
#                 env.final_image = mpimg.imread("Final_Dead.jpg")
#                 env.ax.imshow(env.final_image, extent=[0, env.grid_size, 0, env.grid_size])
#                 plt.draw()
#                 plt.axis("off")
#                 plt.pause(3)
#             break
#     env.close()