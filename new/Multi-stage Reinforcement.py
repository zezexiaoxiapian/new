import numpy as np
import matplotlib.pyplot as plt

class CollaborativeTargetSearch:
    def __init__(self, num_agents, environment_size):
        self.num_agents = num_agents
        self.environment_size = environment_size
        self.agent_positions = [np.random.rand(2) * environment_size for _ in range(num_agents)]
        self.target_positions = [np.random.rand(2) * environment_size for _ in range(3)]
        self.target_velocities = [np.random.rand(2) * 2 - 1 for _ in range(3)]
        self.noise_std_dev = 0.1
        self.boundary = environment_size
        self.total_distance = 0
        self.initial_positions = self.agent_positions.copy()

    def run(self):
        plt.figure(figsize=(15, 10))
        captured_targets = []
        agent_colors = plt.cm.jet(np.linspace(0, 1, self.num_agents))

        while len(captured_targets) < len(self.target_positions):
            plt.clf()

            for t, target_pos in enumerate(self.target_positions):
                if t in captured_targets:
                    continue

                target_pos += np.random.normal(0, self.noise_std_dev, target_pos.shape)
                target_pos += self.target_velocities[t]
                self.target_velocities[t] = self.bounce_object(target_pos, self.target_velocities[t], self.boundary)

                distances = [np.linalg.norm(agent_pos - target_pos) for agent_pos in self.agent_positions]
                closest_agent_idx = np.argmin(distances)

                if distances[closest_agent_idx] < 0.5:
                    captured_targets.append(t)
                    continue

                move_direction = (target_pos - self.agent_positions[closest_agent_idx]) / distances[closest_agent_idx]
                self.agent_positions[closest_agent_idx] += move_direction * 0.1

                self.agent_positions[closest_agent_idx] += np.random.normal(0, self.noise_std_dev, self.agent_positions[
                    closest_agent_idx].shape)
                self.bounce_object(self.agent_positions[closest_agent_idx], [0, 0], self.boundary)

            for i, agent_pos in enumerate(self.agent_positions):
                plt.scatter(*agent_pos, color=agent_colors[i], s=100, alpha=0.6, label=f'Agent {i + 1}')
            for target_pos in self.target_positions:
                plt.scatter(*target_pos, color='b', marker='x', s=150)

            self.total_distance += sum(
                [np.linalg.norm(self.agent_positions[i] - self.initial_positions[i]) for i in range(self.num_agents)])
            self.initial_positions = [pos.copy() for pos in self.agent_positions]  # 使用深度复制

            plt.title('Drones Tracking Multiple Moving Targets with Collaborative Target Search')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.grid(True)
            plt.xlim(0, self.boundary[0])
            plt.ylim(0, self.boundary[1])
            plt.legend()
            plt.pause(0.1)

        print(f"Total distance traveled by all drones (CTS approach): {self.total_distance:.2f} units")
        plt.show()

    def bounce_object(self, pos, vel, boundary):
        for i in range(2):
            if pos[i] <= 0:
                vel[i] = abs(vel[i])
                pos[i] = 0
            elif pos[i] >= boundary[i]:
                vel[i] = -abs(vel[i])
                pos[i] = boundary[i]
        return vel

# Sample parameters
N = 5  # Number of agents
boundary = [20, 20]  # Boundary for the area

cts = CollaborativeTargetSearch(num_agents=N, environment_size=boundary)
cts.run()
