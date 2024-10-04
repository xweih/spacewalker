# The Space Walker
## An Astronaut's Spacewalk Scheduling for Satellite Repairs: A Capacitated TSP with Lateness

<img src="images/spacewalker.png" width="1000" >
( _Image Generated by Copilot_ )


In the realm of modern space exploration, satellites play a critical role in global communications, weather monitoring, GPS, and scientific research. These advanced technologies orbit Earth in precise, synchronized paths, but like any machinery, they are prone to malfunctions. When satellites experience technical issues, timely repairs are essential to prevent service disruptions and data loss. Astronauts stationed on the International Space Station (ISS) are often tasked with performing spacewalks—extravehicular activities (EVA)—to repair these satellites. Each spacewalk is meticulously planned due to the extreme conditions in space and the limited time astronauts can spend outside the ISS. 

Furthermore, travel between satellites and the station is time-consuming, and repairing a satellite requires specialized tools and processes, making the coordination of spacewalks a complex optimization problem. Since each satellite has a critical window for repair before its systems fail, the challenge lies in scheduling multiple spacewalks efficiently, while adhering to time and resource constraints. The importance of minimizing delays and ensuring all satellites are repaired within their operational deadlines is vital for mission success. 

**Scenario Description**

The present task in this project is to help schedule astronauts' spacewalks to repair 7 malfunctioning satellites that are orbiting Earth. The satellites are arranged in a linear orbital path and need to be repaired before they drift too far off course. Each spacewalk team starts from the International Space Station (ISS) (considered as location 0) and can visit multiple satellites in a single spacewalk.

However, spacewalks are extremely energy-consuming, and an astronaut team can only visit and repair up to 4 satellites per spacewalk. After finishing repairs, they must return to the ISS to recharge their oxygen and tools before they can conduct another spacewalk.

**Parameters**

1. The satellites are evenly spaced along their orbital path.
2. It takes 1 minute for the astronaut team to travel between adjacent satellites (or between the ISS and the first satellite).
3. Each satellite repair takes 2 minutes to complete.
4. The satellites must be repaired by specific times (all in Coordinated Universal Time (UTC)) to avoid system failures:
	- Satellite 1: 12:19 AM
	- Satellite 2: 12:19 AM
	- Satellite 3: 12:25 AM
	- Satellite 4: 12:30 AM
	- Satellite 5: 12:20 AM
	- Satellite 6: 12:35 AM
	- Satellite 7: 12:17 AM

5. Each satellite repair must be completed at least 5 minutes before its system failure time.
6. The astronaut team can start their spacewalks at 12:00 AM from the ISS.

**Objective**

The goal is to schedule the astronauts’ spacewalks to minimize the total delay in repairing the satellites while ensuring all repairs are done in time to prevent system failures.

**Constraints**

1. Each spacewalk team can repair no more than 4 satellites per trip.
2. Each satellite must be repaired exactly once across all spacewalks.
3. Each spacewalk must start and end at the ISS.
4. No two spacewalks can overlap; the astronauts must return to the ISS before starting the next spacewalk.


## The Model

**Indicies and Sets:**

Locations: $S=\\{0,..., N \\}$, $s = \\{1,..., N \\}$

Trips: $\Theta =\\{1,..., K \\}$, $\theta =\\{2,..., K \\}$

Satellite: $i, j \in S$

Trip: $k \in \Theta $ 

**Parameters:**

$A$: distance matrix

$D_i$: failure time (deadline) for each satellite

$W$: idle time required before faliure time is reached 

$N$: total number of satellites

$T$: time at which the astronaut team can start their spacewalks (12:00 AM, 0 minutes)
 
$U$: repair time per satellite in minutes

$capacity$: the number of satellites an astronaut can service in each trip


**Decision variables:**

$x_{ijk} \in$ {0,1}: 1, if a walk from satellite i to j occurs in trip k, and 0, if not.  

$y_{ik} \in$ {0,1}: 1, if satellite i is visited in trip k, and 0, if not.  

$u_{ik} $: the starting time of repair work for satellite $i$ during trip $k$.

$C_{k}$: the completion time of trip $k$ (when the astronaut returns to ISS from a trip).

The problem can be modeled as the following MIP.

**The MILP model:**

$$
\begin{align}
	\text{minimize:}	& \text{maximize} \\{0,\ u_{ik} - U + W - D_i \\} 	&\\    
	\text{subject to:} 	& \sum_{j \in s} x_{0jk} = 1, & \forall k \in \Theta 	\\
    				& \sum_{i \in s} x_{i0k} = 1,  &\forall k \in \Theta 	\\
   				& \sum_{j \in s} x_{ijk} = y_{ik}, & \forall i \in s, k \in \Theta	\\
    				& \sum_{i \in s} x_{ijk} = y_{ik},  &\forall j \in s, k \in \Theta	\\
    				& \sum_{k \in \Theta} y_{ik} = 1,  & \forall i \in s 	\\
				& \sum_{i \in s} y_{ik} \leq capacity,  &\forall k \in \Theta  \\
    				& C_{0} = \sum_{i \in S} \sum_{j \in S} A_{ij}x_{ij0} + U \sum_{i \in s} y_{i0} \\
    				& C_{k} = C_{k-1} + \sum_{i \in S} \sum_{j \in S} A_{ij}x_{ijk} + U \sum_{i \in s} y_{ik}, &\forall k \in \theta \\
    				& 0 \leq u_{ik} \leq M * y_{ik},  & \forall i \in s, k \in \Theta   \\
    				& u_{ik} + M * (1- y_{ik}) \geq C_{k-1},  & \forall i \in s, k \in \theta  \\
    				& u_{ik} + M * (1- y_{ik}) \geq C_{k-1} +  A_{0,i},  & \forall i \in s, k \in \theta  \\
    				& u_{i0} + M * (1- y_{i0}) \geq A_{0i}, \quad\qquad & \forall i \in s  \\
    				& u_{ik} + U + A_{ij} \leq u_{jk} + M * (1- x_{ijk}),  & \forall i,j \in s, k \in \Theta   \\
\end{align}
$$

## The Code

The above mathematical model is encoded in Python Jupyter notebook with [CVXPY](https://www.cvxpy.org/) as the solver. Adding the following routine is necessary. 

```javascript
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
```
 
First, I preprocess the data, i.e., the satellites's locational information from a csv file. 

```javascript
df = pd.read_csv('input_ctsp/locations.csv')
sns.scatterplot(data=df, x='X', y='Y'W)
coords = df[['X','Y']].values  # returns a numpy array
travel_times = distance_matrix(coords, coords)

# Showing distance matrix
print('Distance Matrix is:\n')
print(np.round(travel_times, 4))
```
Showing the location of the satellites. 

<img src="images/spacewalker_locations.png" width="1000" >

```javascript
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
for k in range(n_trips):
    constraints += [u[1:,k] >= 0]
    # constraints += [u[1:,k] >= y[1:,k]]

# If a satellite is NOT scheduled during a trip, its starting time remains 0
# If it IS visited, then no constraint to repair starting time.
for k in range(n_trips):
    for i in range( n_satels):
        constraints += [u[i,k] <= M * y[i,k] ] 
        
# for k in range(1, n_trips):
#     for i in range( n_satels):
#         constraints += [u[i,k] + M * (1-y[i,k]) >= C[k-1] ] 

    
# 6. Trip Sequencing Constraints (trips must be sequential in time: no overlapping)

# for i in range(1,  n_satels+1):
#     constraints += [ u[i-1, 0] >= 0 + y[i-1, 0] * travel_times[0, i] ]
for i in range(n_satels):
    constraints += [ u[i, 0] >= 0 + y[i, 0] * travel_times[0, i+1] ]

# 7. Bounds of repair starting time, u[i,k]

# 7.1 starting time must be smaller than the completion time of the same trip 
for k in range(n_trips):
    for i in range(n_satels):
        constraints += [ u[i,k] + U + travel_times[i+1, 0] <= C[k] ]

# 7.2 If a satellite is vited in a trip (k), its repiar starting time must be greater than the previous completion time.
#     If NOT visited, however, no constraint is imposed to the starting time. 
for k in range(1, n_trips):
    for i in range( n_satels):
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
```

## Post-processing

Display the results from the solutions of the optimization problem. 

```javascript
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
```

## Results

After running the script, we are able to find at least one optimal solution, as follows.

```
Optimal total delay: 17.87 minutes

Detailed satellite-repair assignments to trips:

Trip 1:
Satellite 2 completes unloading: 6.47 minutes after 12:00 AM, late by 0.0 minutes.
Satellite 3 completes unloading: 9.89 minutes after 12:00 AM, late by 0.0 minutes.
Satellite 4 completes unloading: 13.30 minutes after 12:00 AM, late by 0.0 minutes.
Satellite 5 completes unloading: 16.71 minutes after 12:00 AM, late by 1.7 minutes.

Trip 2:
Satellite 1 completes unloading: 25.23 minutes after 12:00 AM, late by 2.2 minutes.
Satellite 6 completes unloading: 32.61 minutes after 12:00 AM, late by 7.6 minutes.
Satellite 7 completes unloading: 36.03 minutes after 12:00 AM, late by 14.0 minutes.
```

## Visualization

<img src="images/spacewalker_routes.png" width="1000" >

<img src="images/spacewalker_heatmap.png" width="1000" >
