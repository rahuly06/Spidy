import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_q_table(goal_coordinates=(5, 4), obstacles_coordinates=[(2, 2), (4, 1), (2, 4), (5, 3), (4, 5)], 
                      power_up_coordinates=(4, 2), actions=["Up", "Down", "Left", "Right"], q_values_path="q_table.npy"):

    try:
        q_table = np.load(q_values_path)
        print("Q-table loaded successfully.")                                       # Loading the Q table

        _, axes = plt.subplots(1, 4, figsize=(20, 5))                               # Plots based on action

        for i, action in enumerate(actions):                                        # Iterate action
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()                                  # Extract Q Value for current action

            mask = np.zeros_like(heatmap_data, dtype=bool)                          # Masking special states
            mask[goal_coordinates] = True
            mask[power_up_coordinates] = True
            for obstacle in obstacles_coordinates:
                mask[obstacle] = True

            sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=i == 3, cbar_kws={"orientation": "vertical"}, 
                        mask=mask)                                                  # Plotting heatmap

            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(power_up_coordinates[1] + 0.5, power_up_coordinates[0] + 0.5, 'P', color='yellow',
                    ha='center', va='center', weight='bold', fontsize=14)
            for obstacle in obstacles_coordinates:
                ax.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'O', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
            ax.invert_yaxis()                                                      

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"No saved Q-table found at path: {q_values_path}. Please check your path.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")