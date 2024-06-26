import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('node_aggregated_statistics.csv')

# Extract data for plotting
node_id = df['node_ID']
mean = df['mean']
median = df['median']
mode = df['mode']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(node_id, mean, marker='o', linestyle='-', color='b', label='Mean')
plt.plot(node_id, median, marker='s', linestyle='--', color='g', label='Median')
plt.plot(node_id, mode, marker='^', linestyle=':', color='r', label='Mode')

# Adding labels and title
plt.xlabel('Node ID')
plt.ylabel('Number of Vehicles')
plt.title('Number of Vehicles by Node ID')
plt.xticks(node_id)  # Ensure all node IDs are shown on x-axis
plt.grid(True)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
