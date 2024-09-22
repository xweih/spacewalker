# Spacewalk
Astronaut Spacewalk Scheduling for Satellite Repairs


Scenario Description:
You are responsible for scheduling astronauts' spacewalks to repair 7 malfunctioning satellites that are orbiting Earth. The satellites are arranged in a linear orbital path and need to be repaired before they drift too far off course. Each spacewalk team starts from the International Space Station (ISS) (considered as location 0) and can visit multiple satellites in a single spacewalk.

However, spacewalks are extremely energy-consuming, and an astronaut team can only visit and repair up to 4 satellites per spacewalk. After finishing repairs, they must return to the ISS to recharge their oxygen and tools before they can conduct another spacewalk.

Parameters:
The satellites are evenly spaced along their orbital path.
It takes 1 minute for the astronaut team to travel between adjacent satellites (or between the ISS and the first satellite).
Each satellite repair takes 2 minutes to complete.
The satellites must be repaired by specific times to avoid system failures:
Satellite 1: 9:19 AM
Satellite 2: 9:19 AM
Satellite 3: 9:25 AM
Satellite 4: 9:30 AM
Satellite 5: 9:20 AM
Satellite 6: 9:35 AM
Satellite 7: 9:17 AM
Each satellite repair must be completed at least 5 minutes before its system failure time.
The astronaut team can start their spacewalks at 9:00 AM from the ISS.
Objective:
The goal is to schedule the astronauts’ spacewalks to minimize the total delay in repairing the satellites while ensuring all repairs are done in time to prevent system failures.

Constraints:
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
