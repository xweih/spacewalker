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

$
x_{ijk} = \left\{
    \begin{array}\\
        1 & \text{a walk from satellite } i \text{ to } j \text{ occurs in trip } k\\
        0 & \text{otherwise} \\
    \end{array}
\right.
$

$
y_{ik} = \left\{
    \begin{array}\\
        1 & \text{ satellite } i \text{ is serviced during trip } k,  \\
        0 & \text{otherwise} \\
    \end{array}
\right.
$

$u_{ik} $: the starting time of repair work for satellite $i$ during trip $k$.

$C_{k}$: the completion time of trip $k$ (when the astronaut returns to ISS from a trip).

The problem can be modeled as the following MIP.


$$
\begin{align}
\mbox{Union: } & A\cup B = \{x\mid x\in A \mbox{ or } x\in B\} \\
\mbox{Concatenation: } & A\circ B  = \{xy\mid x\in A \mbox{ and } y\in B\} \\
\mbox{Star: } & A^\star  = \{x_1x_2\ldots x_k \mid  k\geq 0 \mbox{ and each } x_i\in A\} \\
\end{align}
$$

$$
\begin{align*} 
2x - 5y &=  8 \\ 
3x + 9y &=  -12
\end{align*}
$$


