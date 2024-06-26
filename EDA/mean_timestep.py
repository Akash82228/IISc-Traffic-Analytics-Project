import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('timestep_statistics.csv')

# Ensure the 'timestep' column is sorted
df = df.sort_values(by='timestep')

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['timestep'], df['mean'], linestyle='-')

# Adding titles and labels
plt.title('Timestep vs number of vehicles')
plt.xlabel('Timestep')
plt.ylabel('mean number of vehicles')

# Set x-axis range and intervals
plt.xticks(ticks=range(0, 2017, 168), labels=[f'T{i}' for i in range(0, 2017, 168)])

# Show the plot
plt.grid(True)
plt.show()
