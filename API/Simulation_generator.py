#from Cumulative_csv import simulate, create_csv
from Simulation_Functions import get_all_required_paths, generate_weights_files, generate_demand_configuration, generate_demand, delete_all_files, merge_trips, generate_sumocfg
import pandas as pd

def Generate_Sumo_Data(network:str = 'default', veh_distribution = [], simulation_granularity = 60*60 ,usr_paths_dict:dict = {}, usr_demand_configuration:dict = {}):
    
    #SECTION - Get the required Paths
    paths_dict = get_all_required_paths(network, usr_paths_dict)
    print(paths_dict)
    
    # Read the weights csv files
    weights_path = paths_dict['weights_path']
    src_df = pd.read_csv(weights_path['src'])
    dst_df = pd.read_csv(weights_path['dst'])
    via_df = pd.read_csv(weights_path['via'])

    #TODO - Make sure this can be more flexible in terms of weights
    #TODO - Make this more flexible in terms of time
    # Ensure length of source, destination, via and length of distribution is the same. +1 is added to veh_distribution since the rest have an extra column for edge_ID
    if not (len(src_df.columns) == len(dst_df.columns) == len(via_df.columns) == (len(veh_distribution) + 1)): 
        raise Exception("The length of vehicle distribution, source, destination and via do not match.")
    
    # Generate Demand Configuration that will be static
    demand_configuration = generate_demand_configuration(network, usr_demand_configuration)
    print(f"Random Seed Value: {demand_configuration['Random_seed']}")

    #TODO - Delete all trips file
    # Delete Demand files
    delete_all_files(paths_dict['demand_output_path'],'.trips.xml')
    delete_all_files(paths_dict['demand_output_path'],'.rou.xml')

    # Create Multiple Trips based on vehicle distribution over time
    for loop_number, veh_per_hour in enumerate(veh_distribution):

        # Simulation Time
        t_start = loop_number * simulation_granularity
        t_end = (loop_number+1) * simulation_granularity

        source_data = src_df.iloc[:,[0,loop_number+1]].values
        destination_data = dst_df.iloc[:,[0,loop_number+1]].values
        via_data = via_df.iloc[:,[0,loop_number+1]].values

        # Generate Weight Files
        temp_weights_file_path = weights_path['Folder']+'\Weights'
        generate_weights_files(source_data, t_start, t_end, 'src', temp_weights_file_path)
        generate_weights_files(destination_data, t_start, t_end, 'dst', temp_weights_file_path)
        generate_weights_files(via_data, t_start, t_end, 'via', temp_weights_file_path)

        #TODO - Give paths seperately, do not give the entire paths dictionary
        # Generate Demand Data
        temp_trips_file_path, temp_routes_file_path = generate_demand(paths_dict, temp_weights_file_path, demand_configuration, veh_per_hour, t_start=t_start, t_end=t_end)
        # Merge seperate trips into 1 common file, all trips have to have a unique id
        merge_trips(paths_dict['trips_path'],temp_trips_file_path, temp_routes_file_path)

    #TODO - Create a seperate code to generate sumocfg independent of this function
    # Generate Sumocfg file
    generate_sumocfg(paths_dict['network_path'], paths_dict['trips_path'], paths_dict['sumocfg_path'])