import pandas as pd
from statistics import mean, median, mode, StatisticsError

# Load the CSV file
input_file = 'real_data.csv'
output_file = 'node_aggregated_statistics.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Initialize a list to store the results
results = []

# Iterate over each row (each node)
for index, row in df.iterrows():
    node_id = row['node_ID']
    
    # Extract the vehicle counts for all timesteps
    vehicle_counts = row[1:]  # Exclude the first column (node_ID)
    vehicle_counts_list = vehicle_counts.tolist()
    
    # Calculate mean, median, and mode for the vehicle counts
    mean_value = mean(vehicle_counts_list)
    median_value = median(vehicle_counts_list)
    
    try:
        mode_value = mode(vehicle_counts_list)
    except StatisticsError:
        mode_value = "No mode"  # Handle the case where there is no mode
    
    # Append the results to the list
    results.append([node_id, mean_value, median_value, mode_value])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['node_ID', 'mean', 'median', 'mode'])

# Write the results to an output CSV file
results_df.to_csv(output_file, index=False)

print(f"Statistics have been written to {output_file}")
