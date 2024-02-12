import traci
import pandas as pd

# Configuration for starting SUMO simulation with TraCI
sumoCmd = ["sumo", "-c", "/home/akashs/codes/3-06-24/sumocfg_file_try_2.sumocfg"]  # Update this path as necessary

# Start TraCI with the simulation configuration
traci.start(sumoCmd)

# Get a list of all edges in the simulation
edges = traci.edge.getIDList()

# Initialize a dictionary to hold vehicle counts per time step for each edge
edge_data = {edge: [] for edge in edges}

# Simulation loop
step = 0
time_intervals = []
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    if step % 60 == 0:  # Collect data every 300 seconds
        time_intervals.append(f't_{step}')  # Record the current time interval
        for edge in edges:
            vehicles_count = traci.edge.getLastStepVehicleNumber(edgeID=edge)
            edge_data[edge].append(vehicles_count)
        
    step += 1

# Close TraCI connection
traci.close()

# Prepare data for DataFrame
data = {'edge': edges}
data.update({time: [counts[i] for counts in edge_data.values()] for i, time in enumerate(time_intervals)})

# Create DataFrame
df = pd.DataFrame(data)
csv_path = "EdgeVehicles.csv"  # Adjust path as needed
df.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")
