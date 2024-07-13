# Importing all necessary libraries
import torch
import random
from collections import deque
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import torch.nn as nn
import torch.optim as optim

# Replay Buffer:
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

# Train function:
def train(q_net, 
          q_target, 
          memory, 
          optimizer,
          batch_size,
          gamma):
    
    for _ in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q_net(s)

        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Deep Q-Network:
class Qnet(nn.Module):
    def __init__(self, no_actions, no_states):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(no_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, observation, epsilon):
        a = self.forward(observation)
        
        if random.random() < epsilon:
            return random.randint(0, 3)
        else: 
            return a.argmax().item()

# Custom Environment: Spidy
class Spidy(gym.Env):
    def __init__(self, grid_size=6):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([0, 0])
        self.goal_state = np.array([4, 5])  
        self.stacy_health = 100
        self.timer = 60
        self.power_up_state = np.array([2, 4])
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

    def reset(self):
        self.agent_state = np.array([0, 0])
        self.start_time = time.time()
        self.stacy_health = 100
        self.timer = 60
        return self.agent_state

    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size - 1:
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:
            self.agent_state[0] += 1

        done = False
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

        self.stacy_health -= 1
        elapsed_time = time.time() - self.start_time
        self.timer = 60 - int(elapsed_time)

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)

        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.imshow(self.background_img, extent=[-1, self.grid_size, -1, self.grid_size])

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

    def close(self):
        plt.close()

# Main function to train and test DQN on Spidy environment
if __name__ == "__main__":
    train_dqn = True
    test_dqn = False
    render = False

    env = Spidy()

    # Hyperparameters
    no_actions = env.action_space.n
    no_states = env.observation_space.shape[0]
    learning_rate = 0.005
    gamma = 0.98
    buffer_limit = 50000
    batch_size = 32
    num_episodes = 10000
    max_steps = 1000

    if train_dqn:
        q_net = Qnet(no_actions=no_actions, no_states=no_states)
        q_target = Qnet(no_actions=no_actions, no_states=no_states)
        q_target.load_state_dict(q_net.state_dict())

        memory = ReplayBuffer(buffer_limit=buffer_limit)
        optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

        print_interval = 20
        episode_reward = 0.0
        rewards = []

        for n_epi in range(num_episodes):
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
            s = env.reset()
            done = False

            for _ in range(max_steps):
                a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done, info = env.step(a)

                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r, s_prime, done_mask))
                s = s_prime

                episode_reward += r

                if done:
                    break

            if memory.size() > 2000:
                train(q_net, q_target, memory, optimizer, batch_size, gamma)

            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q_net.state_dict())
                print(
                    f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

            rewards.append(episode_reward)
            episode_reward = 0.0

            if rewards[-10:] == [max_steps]*10:
                break

        env.close()
        torch.save(q_net.state_dict(), "dqn_spidy.pth")

        plt.plot(rewards, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.legend()
        plt.savefig("training_curve_spidy.png")
        plt.show()

    if test_dqn:
        print("Testing the trained DQN: ")
        env = Spidy()

        dqn = Qnet(no_actions=no_actions, no_states=no_states)
        dqn.load_state_dict(torch.load("dqn_spidy.pth"))

        for _ in range(10):
            s = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                action = dqn(torch.from_numpy(s).float())
                s_prime, reward, done, info = env.step(action.argmax().item())
                s = s_prime

                episode_reward += reward

                if done:
                    break
            print(f"Episode reward: {episode_reward}")

        env.close()
