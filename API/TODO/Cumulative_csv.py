import traci
import pandas as pd
from Simulation_Functions import parser

def simulate(Config_File_Location, time_granularity):
    # Configuration for starting SUMO simulation with TraCI
    sumoCmd = ["sumo", "-c", Config_File_Location]  # Update this path as necessary

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
        
        # Listen for the "vehicle.teleported" TraCI command and stop the simulation if a vehicle is teleported
        teleported_vehicles = traci.vehicle.getTeleportingIDList()
        if teleported_vehicles:
            raise Exception(f"Vehicle {teleported_vehicles[0]} has been teleported. Stopping simulation.")

        # Accumulate vehicle IDs for each junction at this time step
        for junction in junctions:
            incoming_edges = traci.junction.getIncomingEdges(junctionID=junction)
            for edge_id in incoming_edges:
                # Check if the edge is an internal connection and skip if it is
                if not edge_id.startswith(':'):
                    accumulated_counts[junction].update(traci.edge.getLastStepVehicleIDs(edgeID=edge_id))  # Add the unique vehicle IDs to the set

        if step > 0 and step % time_granularity == 0:
            # Collect the accumulated vehicle counts for each junction
            for junction in junctions:
                junction_data[junction].append(len(accumulated_counts[junction]))  # Get the size of the set and append it to the data
                accumulated_counts[junction].clear()  # Reset the set for the next minute

        step += 1

        if step % 60 != 0:
            for junction in junctions:
                junction_data[junction].append(len(accumulated_counts[junction]))  # Get the size of the set and append it to the data


    # Close TraCI connection
    traci.close()

    return junctions,junction_data

def create_csv(junctions, junction_data, FilePath):

    # Prepare data for DataFrame
    time_steps = [f't_{i}' for i, _ in enumerate(junction_data[next(iter(junction_data))])]
    data = {'node_ID': junctions}
    data.update({time: [counts[i] for counts in junction_data.values()] for i, time in enumerate(time_steps)})

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(FilePath, index=False)

    print(f"Data saved to {FilePath}")