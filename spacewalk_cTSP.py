"""
The Space Walker
An Astronaut's Spacewalk Scheduling for Satellite Repairs: A Capacitated TSP with Lateness

Created on Thursday, Oct 3rd 21:12:47 2024

@author: Xiaowei Hu
"""

################################################
# Loading libraries
################################################
#!pip install --quiet geopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import cvxpy as cp
import math
from scipy.spatial import distance_matrix
# from geopy import distance # Library for geographical calculations


################################################
# Building distance matrix
################################################

df = pd.read_csv('input_ctsp/locations.csv')
sns.scatterplot(data=df, x='X', y='Y')
plt.savefig("spacewalker_locations.png")
coords = df[['X','Y']].values  # returns a numpy array
travel_times = distance_matrix(coords, coords)

# Showing distance matrix
print('Distance Matrix is:\n')
print(np.round(travel_times, 4))

################################################
# Prepare the input
################################################

# Indices
n_satels = len(df) - 1
capacity = 4
n_trips = math.ceil( n_satels / capacity)
satellites = np.arange(1, n_satels + 1)

# Parameters
# Departure times in minutes after 12:00 AM
#failure_times = np.array([19, 19, 25, 30, 20, 35, 17])  
# failure_times = np.array([34, 34, 34, 34, 34, 34, 30]) 
rng = np.random.default_rng(seed=12345)
failure_times = rng.integers(19, 36, size =  n_satels)

# Time at which the astronaut team can start their spacewalks (12:00 AM, 0 minutes)
T = 0  

# Repair time per satellite in minutes
U = 2  

M = 100000  # Big M

################################################
# Solving the integer programming problem
################################################

# Defining the decision variables

# x_ijk: Binary variable for each trip, dim = 8*8*2
x = [cp.Variable(( n_satels + 1,  n_satels + 1), boolean=True) for _ in range(n_trips)] 

# u_ik: Start time of repair at satellite i during trip k, dim = 7*2
u = cp.Variable(( n_satels, n_trips), nonneg=True)  

# y_ik: Whether satellite i is serviced during trip k, dim = 7 *2
y = cp.Variable(( n_satels, n_trips), boolean=True)  

# Completion time of each trip
C = cp.Variable(n_trips, nonneg=True)  

# Objective function: Minimize total delay
delay = cp.sum(cp.maximum(0, u + U - (failure_times.reshape(-1, 1) - 5)))
objective = cp.Minimize(delay)


# Constraints
constraints = []

# 0. Initial constraintsx_ii = 0 
for k in range(n_trips):
    constraints += [cp.diag(x[k]) == 0]

# 1. Flow conservation
for k in range(n_trips):
    # Start at International Space Station (satellite 0) and must visit each assigned satellite once
    constraints += [ cp.sum(x[k][0, 1:]) == 1 ]
    constraints += [ cp.sum(x[k][1:, 0]) == 1 ]    
    # Each satellite assigned to a trip must have exactly one preceding and one succeeding satellite
    for i in range(1,  n_satels + 1):
        constraints += [ cp.sum(x[k][i, :]) == y[i - 1, k] ]
        constraints += [ cp.sum(x[k][:, i]) == y[i - 1, k] ]

# 2. Assignment constraints: Each satellite must be served exactly once across all trips
for i in range(n_satels):
    constraints += [ cp.sum(y[i, :]) == 1 ]

# 3. Spaceman capacity constraints: No more than 4 satellites in each trip
for k in range(n_trips):
    constraints += [ cp.sum(y[:, k]) <= capacity ]

# 4. Definition of Completion Time, C[k]
constraints += [ C[0] == 0 + cp.sum( cp.multiply(travel_times, x[0]) ) + U * cp.sum(y[:,0]) ]

for k in range(1, n_trips):
    constraints += [ C[k] == C[k-1] + cp.sum( cp.multiply(travel_times, x[k]) )  + U * cp.sum(y[:,k]) ]


# 5. Repair starting time
# for k in range(n_trips):
#     constraints += [u[:,k] >= 0]
    # constraints += [u[1:,k] >= y[1:,k]]
constraints += [u[:,:] >= 0]

# If a satellite is NOT scheduled during a trip, its starting time remains 0
# If it IS visited, then no constraint to repair starting time.
for k in range(n_trips):
    for i in range( n_satels):
        constraints += [u[i,k] <= M * y[i,k] ] 

# This constraint is absorbed by 7.2        
# for k in range(1, n_trips):
#     for i in range( n_satels):
#         constraints += [u[i,k] + M * (1-y[i,k]) >= C[k-1] ] 

    
# 6. [obsolete] Trip Sequencing Constraints (trips must be sequential in time: no overlapping)

# The following may be the same as 7.2
# for i in range(n_satels):
#     constraints += [ u[i, 0] >= 0 + y[i,0] * travel_times[0, i+1] ]

# [caution] The following seemingly true constraint CANNOT hold, as u[i, k] need to be 0 sometimes.
# for k in range(n_trips):
#     for i in range(n_satels):
#         constraints += [ u[i, k] >= C[k-1] + y[i,k] * travel_times[0, i+1] ]
        
# 7. Bounds of repair starting time, u[i,k]

# 7.1 Upper bound: starting time (returning to ISS directly) must be smaller than the completion time of the same trip 
for k in range(n_trips):
    for i in range(n_satels):
        constraints += [ u[i,k] + U + travel_times[i+1, 0] <= C[k] ]

# 7.2 Lower bound: If a satellite is vited in a trip (k), 
#     its repiar starting time () must be greater than the previous completion time.
#     If NOT visited, however, no constraint is imposed to the starting time. 
for k in range(1, n_trips):
    for i in range(n_satels):
        constraints += [ C[k-1] + travel_times[0, i+1] <= u[i,k] + M * (1 - y[i,k]) ]

for i in range(n_satels):
    constraints += [ travel_times[0, i+1] <= u[i,0] + M * (1-y[i,0]) ]

# 8. Subtour elimination constraints
for k in range(n_trips):
    for i in range(1,  n_satels + 1):
        for j in range(1,  n_satels + 1):
            if i != j:
                constraints += [u[i - 1, k] + U + travel_times[i, j] <= u[j - 1, k] + M * (1 - x[k][i, j]) ]

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(verbose=False)

################################################
# Results
################################################

if y.value is not None:
    print(f"Optimal total delay: {prob.value:.2f} minutes")
    print()
    print("Detailed satellite-repair assignments to trips:")
    print()
    
    trip_routes = []
    delayTable = np.zeros((n_satels, n_trips))
    
    for k in range(n_trips):
        print(f"Trip {k + 1}:")
        
        # Extract the route for trip k based on x[k]
        route = []
        current_satellite = 0  # Start from the International Space Station (satellite 0)
        
        while True:
            next_satellite = np.argmax(x[k].value[current_satellite, :])  # Find the next satellite
            if next_satellite == 0:
                break  # End the route when the truck returns to satellite 0 (ISS)
            route.append(next_satellite)  # Add the gate to the route
            current_satellite = next_satellite  # Move to the next satellite
            
        # Print the route in the order the satellites are visited
        for satellite in route:
            unload_time = u.value[satellite - 1, k] + U
            delay = max(0, unload_time + U + 5 - failure_times[satellite - 1])
            delayTable[satellite-1, k] = delay
            print(f"Satellite {satellite} completes unloading: {unload_time:.2f} minutes after 12:00 AM, late by {delay:.1f} minutes.")
        
        #print(route)
        excursion = route
        excursion.insert(0,0)
        excursion.append(0)
        trip_routes.append(excursion)
        #print(excursion)
        #print(trip_routes)
        print()
        
else:
    print("No feasible solution found.")
    
    
    ###################################
    # Network route map
    ###################################
    
    # Create a directed graph
G = nx.DiGraph()

# Add nodes for the satellites and the International Space Station (ISS)
# n_gates = n_satels
G.add_node(0, pos=(0, 0), label="ISS (0)")  # ISS

# for i in range(1, n_satels + 1):
#     G.add_node(i, pos=(i, 0), label=f"Satellite {i}")

for i in range(1, n_satels + 1):
    G.add_node(i, pos=tuple(coords[i]), label=f"Satellite {i}")
    

# Example routes for two trips (from Gantt chart data)
# trip_routes = [
#     [0, 1, 2, 3, 0],  # Trip 1: ISS -> Satellite 1 -> Satellite 2 -> Satellite 3 -> ISS
#     [0, 4, 5, 6, 7, 0]  # Trip 2: ISS -> Satellite 4 -> Satellite 5 -> Satellite 6 -> Satellite 7 -> ISS
# ]

# Add edges based on the route
for route in trip_routes:
    for i in range(len(route) - 1):
        G.add_edge(route[i], route[i + 1])

# Get positions for nodes
pos = nx.get_node_attributes(G, 'pos')

# Plot the graph
plt.figure(figsize=(8, 6))

# Draw nodes with labels
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)

# Draw the edges with arrows representing the trips
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrowstyle='-|>', arrowsize=50, width=2.0)

# Set the title and display the route map
plt.title('Route Map: Astronaut Trips to Satellites')
plt.axis('off')
plt.savefig("spacewalker_routes.png")
plt.show()


  ###################################
  # Heatmap
  ###################################
  
  # Sample delay data for two trips (in minutes)
# Each row represents a satellite (gate) and each column represents a trip
# For simplicity, you can input the actual delays computed in your solution
# delays = np.array([
#     [2, 0],  # Satellite 1: 2 minutes delay in Trip 1, 0 minutes in Trip 2
#     [3, 0],  # Satellite 2
#     [1, 0],  # Satellite 3
#     [0, 5],  # Satellite 4
#     [0, 2],  # Satellite 5
#     [0, 3],  # Satellite 6
#     [0, 1],  # Satellite 7
# ])

# Define labels for the satellites (gates) and trips
satellites = [f"Satellite {i}" for i in range(1, n_satels+1)]  # ["Satellite 1", ..., "Satellite 7"]
trips = [f"Trip {i}" for i in range(1, n_trips+1)]  # ["Trip 1", "Trip 2"]

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(delayTable, annot=False, cmap="crest", xticklabels=trips, yticklabels=satellites, cbar_kws={'label': 'Delay (minutes)'})

# Add labels and title
plt.title('Delay Heatmap for Satellite Services')

plt.savefig("spacewalker_heatmap.png")

#plt.xlabel('Trips')
#plt.ylabel('Satellites')

# Display the heatmap
plt.show()