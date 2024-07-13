import numpy as np
from env_ray9209 import Spidy

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, 
                     gamma, q_table_save_path="q_table.npy", render_every=5_000):
    
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))                  # Initialize Q table

    for episode in range(no_episodes):  
        state = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:                                                  # Using random value to explore or exploit
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)                               # Taking new paprameters based on action
            next_state = tuple(next_state)
            total_reward += reward                                                          # Accumulating rewards

            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])     # Implemeting Q learning rule

            state = next_state

            #if episode % render_every == 0:
            #    env.render()                                                                # Render only few episodes

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)                                 # Explore >>> Expliot 
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")