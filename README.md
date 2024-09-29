# Spacewalk
## Astronaut Spacewalk Scheduling for Satellite Repairs: A Capacitated TSP with Lateness

<img src="images/spacewalker.png" width="1000" >
(Image Generated by Copilot)


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

Locations: $S=\{0,..., N \}$, $s=\{1,..., N \}$
Trips: $\Theta =\{0,..., K \}$, $\theta =\{1,..., K \}$

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
	\text{minimize:}	& \text{maximize:} \\{0,\ u_{ik} - U + W - D_i \\} 			& \tag{B}\label{eq:b} \\    
	\text{subject to:} 	& \sum_{j \in S} x_{ijk} = 1, 						& \forall i,k 	\\    
     				& \sum_{i \in S} x_{ijk} = 1,  						&\forall j,k  \\
    				& \sum_{k \in \Theta} y_{ik} = 1,  					&\forall i  \\
				& \sum_{i \in S} y_{ik} \leq capacity,  				&\forall k   \\
    				& C_{0} = \sum_{i \in S} \sum_{j \in S} A_{ij}x_{ij0} + U \sum_{i \in s} y_{i0} & \\
   				& C_{k} = C_{k-1} + \sum_{i \in S} \sum_{j \in S} A_{ij}x_{ijk} + U \sum_{i \in s} y_{ik},  &\forall k \in \theta \\
				& u_{ik} \geq 0, \quad\qquad &\forall i \in s, k \in \theta   \\
    				& u_{ik} \leq M * y_{ik}, \quad\qquad & \forall i \in s, k \in \theta   \\
    				& u_{i0} \geq y_{i0} * A_{ij}, & \forall i \in s,  \\
    				& u_{ik} + U + A_{i0} \leq C_k, \quad\qquad & \forall i \in s, k \in \theta   \\
    				& u_{ik} + M * (1- y_{ik}) \geq C_{k-1} +  A_{0,i}, \quad\qquad & \forall i \in s, k \in \theta   \\
    				& u_{i0} + M * (1- y_{i0}) \geq A_{0,i}, \quad\qquad & \forall i \in s,   \\
    				& u_{ik} + U + A_{ij} \leq u_{jk} + M * (1- x_{ijk}), \quad\qquad & \forall i,j \in s, k \in \theta   \\
\end{align}
$$


