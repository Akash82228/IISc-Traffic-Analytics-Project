import pandas as pd
from statistics import mean, median, mode, StatisticsError


input_file = 'real_data.csv'
output_file = 'timestep_statistics.csv'


df = pd.read_csv(input_file)


results = []


timesteps = df.columns[1:]  # Exclude the first column (node_ID)
for timestep in timesteps:
    # Extract the vehicle counts for the current timestep across all nodes
    vehicle_counts = df[timestep].tolist()
    
    # Calculate mean, median, and mode for the vehicle counts
    mean_value = mean(vehicle_counts)
    median_value = median(vehicle_counts)
    
    try:
        mode_value = mode(vehicle_counts)
    except StatisticsError:
        mode_value = "No mode"  # Handle the case where there is no mode
    
    # Append the results to the list
    results.append([timestep, mean_value, median_value, mode_value])


results_df = pd.DataFrame(results, columns=['timestep', 'mean', 'median', 'mode'])


results_df.to_csv(output_file, index=False)

print(f"Statistics have been written to {output_file}")
