import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import heapq
import math
import LB_optimizer as LB


def custom_astar(G, start, goal, moves, diagonal_cost, robot_radius, obstacles, obstacle_cost):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path

        for dx, dy, move_cost in moves:
            neighbor = (current[0] + dx, current[1] + dy)



            tentative_g_score = g_score[current] + move_cost  # Use the move_cost for diagonal moves

            # Add cost based on obstacle size
            for obs_x, obs_y, obs_radius in obstacles:
                distance = math.sqrt((neighbor[0] - obs_x) ** 2 + (neighbor[1] - obs_y) ** 2)
                if distance < robot_radius + obs_radius:
                    tentative_g_score += obstacle_cost  # Add obstacle cost

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


#sim_time = 30
#step_time = 0.1
#N = int(sim_time / step_time) 




grid_size = (100, 100)
robot_start = (5, 5)
robot_goal = (90, 90)
robot_radius = 3
obs_rad = 10
obstacle = [(50, 50, obs_rad)]
obstacle_cost = 1


#for i in range(1,N):
    #Goal_dest = robot_goal - 



# Create a graph representation of the grid map
G = nx.grid_2d_graph(*grid_size)



# Add nodes corresponding to obstacles with a high cost
for obs in obstacle:
    for node in list(G.nodes):
        if (
            (node[0] - obs[0]) ** 2 + (node[1] - obs[1]) ** 2
            <= obs_rad ** 2
        ):
            G.nodes[node]['cost'] = obstacle_cost

diagonal_cost = 1  # Cost for diagonal moves
# Define possible moves including diagonals and their corresponding costs
moves = [(0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), (1, 1, diagonal_cost), (1, -1, diagonal_cost), (-1, 1, diagonal_cost), (-1, -1, diagonal_cost)]


# Usage
a_star_path = custom_astar(G, robot_start, robot_goal, moves, diagonal_cost, robot_radius, obstacle, obstacle_cost)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, grid_size[0])
ax.set_ylim(0, grid_size[1])

# Plot the obstacle(s)
for obs in obstacle:
    ax.add_patch(plt.Circle((obs[0] , obs[1]), obs_rad, color='red'))

# Plot the start, goal, and A* path
ax.plot(*robot_start, 'go', markersize=10, label='Start')
ax.plot(*robot_goal, 'bo', markersize=10, label='Goal')
line, = ax.plot([], [], 'b-', label='Robot Path')
ax.legend()

"""ax.add_patch(plt.Circle(robot_start, robot_radius, color='green'))
ax.add_patch(plt.Circle(robot_goal, robot_radius, color='red'))
ax.add_patch(plt.Circle(obstacle, obstacle_radius, color='black'))"""

ax.set_title('LB-SGD')

def update(frame):
    if frame < len(a_star_path):
        x, y = zip(*a_star_path[:frame + 1])  # Use `zip` to unpack the points
        line.set_data(x, y)
    return line,


ani = animation.FuncAnimation(fig, update, frames=len(a_star_path) + 1, blit=True, repeat=False)
plt.show()
