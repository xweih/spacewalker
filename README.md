# Spacewalk
Astronaut Spacewalk Scheduling for Satellite Repairs

<img src="images/spacewalker.png" width="1000" >
(Image Generated by Copilot)
<p style="text-align:center;">Sample text with center alignment</p>


In the realm of modern space exploration, satellites play a critical role in global communications, weather monitoring, GPS, and scientific research. These advanced technologies orbit Earth in precise, synchronized paths, but like any machinery, they are prone to malfunctions. When satellites experience technical issues, timely repairs are essential to prevent service disruptions and data loss. Astronauts stationed on the International Space Station (ISS) are often tasked with performing spacewalks—extravehicular activities (EVA)—to repair these satellites. Each spacewalk is meticulously planned due to the extreme conditions in space and the limited time astronauts can spend outside the ISS. Furthermore, travel between satellites and the station is time-consuming, and repairing a satellite requires specialized tools and processes, making the coordination of spacewalks a complex optimization problem. Since each satellite has a critical window for repair before its systems fail, the challenge lies in scheduling multiple spacewalks efficiently, while adhering to time and resource constraints. The importance of minimizing delays and ensuring all satellites are repaired within their operational deadlines is vital for mission success. 

**Scenario Description:**

You are responsible for scheduling astronauts' spacewalks to repair 7 malfunctioning satellites that are orbiting Earth. The satellites are arranged in a linear orbital path and need to be repaired before they drift too far off course. Each spacewalk team starts from the International Space Station (ISS) (considered as location 0) and can visit multiple satellites in a single spacewalk.

However, spacewalks are extremely energy-consuming, and an astronaut team can only visit and repair up to 4 satellites per spacewalk. After finishing repairs, they must return to the ISS to recharge their oxygen and tools before they can conduct another spacewalk.

**Parameters:**

The satellites are evenly spaced along their orbital path.

It takes 1 minute for the astronaut team to travel between adjacent satellites (or between the ISS and the first satellite).

Each satellite repair takes 2 minutes to complete.

The satellites must be repaired by specific times (all in Coordinated Universal Time (UTC)) to avoid system failures:

Satellite 1: 12:19 AM
Satellite 2: 12:19 AM
Satellite 3: 12:25 AM
Satellite 4: 12:30 AM
Satellite 5: 12:20 AM
Satellite 6: 12:35 AM
Satellite 7: 12:17 AM

Each satellite repair must be completed at least 5 minutes before its system failure time.

The astronaut team can start their spacewalks at 9:00 AM from the ISS.

**Objective:**

The goal is to schedule the astronauts’ spacewalks to minimize the total delay in repairing the satellites while ensuring all repairs are done in time to prevent system failures.

**Constraints:**

Each spacewalk team can repair no more than 4 satellites per trip.
Each satellite must be repaired exactly once across all spacewalks.
Each spacewalk must start and end at the ISS.
No two spacewalks can overlap; the astronauts must return to the ISS before starting the next spacewalk.
Mapping to the Original Problem:
Gates in the original problem are now satellites.
Bags become satellite repair tasks.
The bag runner is the astronaut spacewalk team.
Departure times become the satellite system failure times.
Unloading time (2 minutes) is now the time it takes to repair a satellite.
The central bag room is the International Space Station (ISS).

Unique Twist:
This scenario takes you to outer space and involves astronauts performing critical spacewalks to save malfunctioning satellites. The urgency of preventing satellite system failures adds a high-stakes element, while the idea of scheduling spacewalks creates a vivid and exciting context that’s entirely different from delivery-based problems.

Would you like to implement this spacewalk scheduling scenario using the MIP model we developed?
