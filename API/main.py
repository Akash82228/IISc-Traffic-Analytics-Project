from Simulation_generator import Generate_Sumo_Data

# Detwrmines the vehicle distribution, combined with simulation granularity you can determine number of vehicles per simulation granularity. 
# Ex, veh = [200,300] and simulation granularity = 3600. 200 vehicles will run in the first hour and 300 vehicles will run in the second hour

veh_distribution = [240,240,240,240,240,600,600,600,1800,1800,1800,1200,1200,1200,1200,1200,1200,1800,1800,1800,600,600,600,240]
simulation_granularity = 3600 # Determines the time duration of each simulation
#usr_demand_configuration = {'intermediate':'0'}

Generate_Sumo_Data(network = 'iisc', veh_distribution=veh_distribution)