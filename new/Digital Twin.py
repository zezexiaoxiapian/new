import numpy as np
import matplotlib.pyplot as plt

# Define the dynamics of the agent
def agent_dynamics(p, v, u, d):
    p_dot = v
    v_dot = u + d
    return p_dot, v_dot

# k-WTA strategy
def k_WTA(e, k, covariances):
    weights = [np.exp(-0.5 * np.trace(cov)) for cov in covariances]
    weighted_errors = [e[i] * weights[i] for i in range(len(e))]
    sorted_indices = sorted(range(len(weighted_errors)), key=lambda i: weighted_errors[i])
    w = [0] * len(e)
    for i in sorted_indices[:k]:
        w[i] = 1
    return w

# Bounce the object if it hits the boundary
def bounce_object(pos, vel, boundary):
    for i in range(2):
        if pos[i] <= 0:
            vel[i] = abs(vel[i])
            pos[i] = 0
        elif pos[i] >= boundary[i]:
            vel[i] = -abs(vel[i])
            pos[i] = boundary[i]
    return vel


def digital_twin_auction(p, target_positions, N, num_targets):
    # For simplicity, we'll use the distance to the target as the bidding criterion
    bids = np.zeros((N, num_targets))
    for i in range(N):
        for t in range(num_targets):
            bids[i, t] = np.linalg.norm(p[i] - target_positions[t])

    # Assign tasks to drones based on the lowest bid (i.e., shortest distance)
    task_assignments = np.argmin(bids, axis=0)

    # Calculate control signals for each drone based on task assignments
    u = np.zeros((N, 2))
    for i in range(N):
        assigned_task = np.where(task_assignments == i)[0]
        if assigned_task.size > 0:
            u[i] = 0.5*(target_positions[assigned_task[0]] - p[i])  # Doubled the control signal strength

    return u
class DroneEKF:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = np.eye(len(initial_state))
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 定义状态转移矩阵
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 定义控制输入矩阵
        self.B = np.array([
            [0.5, 0],
            [0, 0.5],
            [1, 0],
            [0, 1]
        ])

        # 定义过程噪声协方差矩阵
        self.Q = np.eye(4) * self.process_noise
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 定义测量噪声协方差矩阵
        self.R = np.eye(2) * self.measurement_noise

    def predict(self, control_input):
        self.state = np.dot(self.F, self.state) + np.dot(self.B, control_input)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement, shared_estimates=None):
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        self.covariance = self.covariance - np.dot(np.dot(K, self.H), self.covariance)
        if shared_estimates is not None:
            all_states = [est[0] for est in shared_estimates]
            average_state = np.mean(all_states, axis=0)
            self.state = (self.state + average_state) / 2

# Sample parameters
N = 5  # Number of agents
k = 2  # Number of winners for each target (reduced to 2)
num_targets = 3  # Number of moving targets
boundary = [50, 50]  # Boundary for the area
noise_std_dev = 0.1  # Standard deviation of the noise
capture_distance_threshold = 1.0
drone_speed = 2.0  # Example speed for drones
target_speed = 0.2  # Example speed for targets
capture_distance =1 # Distance within which drone slows down

# Initial positions of agents
p = np.random.rand(N, 2) * 20
initial_positions = p.copy()

# Initial velocities of agents
v = np.zeros((N, 2))

# Control signals
u = np.zeros((N, 2))

# Disturbances
d = np.zeros((N, 2))

# Moving targets
target_positions = np.random.rand(num_targets, 2) * 20
angles = 2 * np.pi * np.random.rand(num_targets)
target_velocities = np.array([[target_speed * np.cos(angle), target_speed * np.sin(angle)] for angle in angles])

# Initialize EKF for each drone
ekf_drones = [DroneEKF(initial_state=np.zeros(4), process_noise=0.1, measurement_noise=0.1) for _ in range(N)]

plt.figure(figsize=(15, 10))
agent_colors = plt.cm.jet(np.linspace(0, 1, N))  # 为每个无人机分配一个颜色

captured_targets = []
total_distance = 0
while len(captured_targets) < num_targets:
    if not plt.get_fignums():
        break

    plt.clf()

    shared_estimates = []
    for drone in ekf_drones:
        shared_estimates.append((drone.state, drone.covariance))

    # 使用Digital Twin (DT)算法替代k-WTA策略
    u = digital_twin_auction(p, target_positions, N, num_targets)

    for t in range(num_targets):
        if t in captured_targets:
            continue
        # Add noise to target position
        target_positions[t] += np.random.normal(0, noise_std_dev, target_positions[t].shape)

        # Update target position
        target_positions[t] += target_velocities[t]

        # Bounce the target if it hits the boundary
        target_velocities[t] = bounce_object(target_positions[t], target_velocities[t], boundary)

        for i in range(N):
            p_dot, v_dot = agent_dynamics(p[i], v[i], u[i], d[i])
            p[i] += p_dot * 0.1
            v[i] += v_dot * 0.1

            # Add noise to drone position
            p[i] += np.random.normal(0, noise_std_dev, p[i].shape)

            # Bounce the drone if it hits the boundary
            v[i] = bounce_object(p[i], v[i], boundary)

            # Update EKF with measurements and shared estimates
            measurement = np.array([p[i][0], p[i][1]])  # Example measurement
            ekf_drones[i].predict(control_input=u[i])
            ekf_drones[i].update(measurement, shared_estimates)

            # Use updated state for control
            updated_state = ekf_drones[i].state
            # Modify control signals based on updated_state
            distance_to_target = np.linalg.norm(p[i] - target_positions[t])
            if distance_to_target < capture_distance_threshold:
                captured_targets.append(t)
                break  # 当目标被一个无人机捕捉后，跳出内部循环
            # Plotting for each agent after it updates
            plt.scatter(p[i, 0], p[i, 1], color=agent_colors[i], s=100, alpha=0.6,
                        label=f'Agent {i + 1}' if t == 0 else "")
            plt.scatter(target_positions[t, 0], target_positions[t, 1], color='b', marker='x', s=150)

        total_distance += sum([np.linalg.norm(p[i] - initial_positions[i]) for i in range(N)])
        initial_positions = p.copy()

        plt.title('Drones Tracking Multiple Moving Targets with Cooperative EKF')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.xlim(0, boundary[0])
        plt.ylim(0, boundary[1])
        if len(captured_targets) == 0:  # 只在第一次迭代时显示图例
            plt.legend()
        plt.pause(0.1)

print(f"Total distance traveled by all drones: {total_distance:.2f} units")

plt.show()
