import traci
import pandas as pd

def count_vehicles_on_junction(junction_id):
    """Counts vehicles on incoming lanes to a junction, ignoring internal connections."""
    incoming_edges = traci.junction.getIncomingEdges(junctionID=junction_id)
    vehicles_count = set()  # Use a set to store unique vehicle IDs
    for edge_id in incoming_edges:
        # Check if the edge is an internal connection and skip if it is
        if not edge_id.startswith(':'):
            vehicles_count.update(traci.edge.getLastStepVehicleIDs(edgeID=edge_id))  # Add the unique vehicle IDs to the set
    return len(vehicles_count)  # Return the number of unique vehicle IDs in the set

# Configuration for starting SUMO simulation with TraCI
sumoCmd = ["sumo", "-c", "E:\Road_Simulation_Project\API\iisc.sumocfg"]  # Update this path as necessary

# Start TraCI with the simulation configuration
traci.start(sumoCmd)

# Get a list of all junctions/nodes in the simulation, filter out any internal junctions
junctions = [j for j in traci.junction.getIDList() if not j.startswith(':')]

# Initialize a dictionary to hold vehicle counts per time step for each junction
junction_data = {junction: [] for junction in junctions}

# Initialize a dictionary to accumulate vehicle counts for each junction over 60 seconds
accumulated_counts = {junction: set() for junction in junctions}

# Simulation loop
step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    # Accumulate vehicle IDs for each junction at this time step
    for junction in junctions:
        incoming_edges = traci.junction.getIncomingEdges(junctionID=junction)
        for edge_id in incoming_edges:
            # Check if the edge is an internal connection and skip if it is
            if not edge_id.startswith(':'):
                accumulated_counts[junction].update(traci.edge.getLastStepVehicleIDs(edgeID=edge_id))  # Add the unique vehicle IDs to the set

    if step > 0 and step % 300 == 0:  # Every 60 steps (1 minute)
        # Collect the accumulated vehicle counts for each junction
        for junction in junctions:
            junction_data[junction].append(len(accumulated_counts[junction]))  # Get the size of the set and append it to the data
            accumulated_counts[junction].clear()  # Reset the set for the next minute        

    step += 1

# Collect the data for the last incomplete minute if any
if step % 300 != 0:
    for junction in junctions:
        junction_data[junction].append(len(accumulated_counts[junction]))  # Get the size of the set and append it to the data

# Close TraCI connection
traci.close()

# Prepare data for DataFrame
time_steps = [f't_{i}' for i, _ in enumerate(junction_data[next(iter(junction_data))])]
data = {'node_ID': junctions}
data.update({time: [counts[i] for counts in junction_data.values()] for i, time in enumerate(time_steps)})

# Create DataFrame
df = pd.DataFrame(data)
csv_path = "E:\Road_Simulation_Project\API\Cumulative_Vehicle_Count\IISC.csv"  # Adjust path as needed
df.to_csv(csv_path, index=False)

print(f"Data saved to {csv_path}")