import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Function to aggregate data to minute-level timesteps
def aggregate_data_to_minute(csv_file):
    df = pd.read_csv(csv_file)
    df.set_index('node_ID', inplace=True)
    minute_df = pd.DataFrame(index=df.index)
    for i in range(60):
        minute_df[f't_{i//60}'] = df[f't_{i}'].add(df[f't_{i+1}'], fill_value=0)
    return minute_df

# Function to generate graph for a node
def generate_graph_for_node(node, minute_df, graph=nx.Graph()):
    # Find the index of the row for the given node
    node_index = minute_df[minute_df.index == str(node)].index[0]

    # Get the row for the given node
    node_row = minute_df.loc[node_index]

    # Add the node and its connections to the graph
    for column in node_row.index:
        connected_node = int(column.split('_')[-1])
        weight = node_row[column]
        if weight != 0:
            graph.add_node(node)
            graph.add_node(connected_node)
            graph.add_edge(node, connected_node, weight=weight)

    # Draw the graph
    nx.draw(graph, with_labels=True)
    plt.show()

# Load your csv file
df = pd.read_csv('your_file.csv')

# Aggregate the data to minute-level timesteps
minute_df = aggregate_data_to_minute('Cumulative_Vehicle_Count\IISC.csv')

# Use the function to generate a graph for a specific node
generate_graph_for_node('10001174201', minute_df)
