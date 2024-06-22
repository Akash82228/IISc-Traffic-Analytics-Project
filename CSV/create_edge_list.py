import xml.etree.ElementTree as ET
import csv

def extract_edge_list(net_xml_path, output_csv_path):
    # Load and parse the SUMO network XML file
    tree = ET.parse(net_xml_path)
    root = tree.getroot()
    
    # Open a CSV file to write the edge list
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['edge', 'source', 'target'])  # Updated header for CSV file

        # Iterate through each edge in the network XML
        for edge in root.findall('edge'):
            # Some edges are internal and do not have 'from' and 'to' attributes
            if 'from' in edge.attrib and 'to' in edge.attrib:
                edge_id = edge.get('id') # NOTE: If you dont want to store the edge_id comment this line
                source = edge.get('from')
                target = edge.get('to')
                writer.writerow([edge_id, source, target])
                # writer.writerow([source, target]) # NOTE: If you dont want to store the edge_id


# Specify the path to your SUMO network XML file and output CSV file
net_xml_path = '/home/akashs/codes/13-06-24/Cubbon.net.xml'  # Update this path
output_csv_path = '/home/akashs/codes/node_level_ST_GAT/sumo_dataset/edge_list.csv'  # Output CSV file path

# Call the function with your file paths

extract_edge_list(net_xml_path, output_csv_path)
