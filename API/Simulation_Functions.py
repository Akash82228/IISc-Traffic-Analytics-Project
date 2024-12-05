import os
import sys
import configparser
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import random
import tempfile

# Read paths.ini file this location holds all the default paths
config = configparser.ConfigParser()
config.read('paths.ini')
sys.path.append(os.path.abspath(config.get('Sumo Path','Tools')))
import randomTrips

#SECTION - Extras

def delete_files(file_path): 
    try:
        os.remove(file_path)
        print(f"Successfully deleted {file_path}")
    except Exception as e:
        raise Exception(f"Failed to delete {file_path}. Reason: {e}")

def delete_all_files(directory,file_type):
    for filename in os.listdir(directory):
        if filename.endswith(file_type):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Successfully deleted {file_path}")
            except Exception as e:
                raise Exception(f"Failed to delete {file_path}. Reason: {e}")
            
#SECTION - Get the required Paths
    
def get_all_required_paths(network, user_dict):
    
    #TODO - Do not allow people to change network path
    #TODO - Store default values for my network
    match network:
        case 'iisc':
            paths_dict = {'network_path' : config.get('Network Path','iisc'),
                          'weights_path' : {'Folder' : config.get('Weights Path','iisc'), 'src' : config.get('Weights Path','iisc src'), 'dst' : config.get('Weights Path','iisc dst'), 'via' : config.get('Weights Path','iisc via')},
                          'demand_output_path' : config.get('Demand Output Path','iisc'),
                          'trips_path' : config.get('Demand Output Path','iisc trips'),
                          'additional_general_path' : config.get('Additional File Path','general'),
                          'sumocfg_path' : config.get('Sumocfg Path','iisc'),
                    }
                    
        case 'cubbon':
            paths_dict = {'network_path' : config.get('Network Path','cubbon'),
                          'weights_path' : {'Folder' : config.get('Weights Path','cubbon'), 'src' :config.get('Weights Path','cubbon src'), 'dst' : config.get('Weights Path','cubbon dst'), 'via' : config.get('Weights Path','cubbon via')},
                          'demand_output_path' : config.get('Demand Output Path','cubbon'),
                          'additional_general_path' :config.get('Additional File Path','general'),
                          'trips_path': config.get('Demand Output Path','cubbon trips'),
                          'sumocfg_path' : config.get('Sumocfg Path','cubbon'),
                    }

        case 'my network':
            paths_dict = {'network_path' : '',
                          'weights_path' : {'Folder' : '', 'src':'', 'dst':'','via':''},
                          'demand_output_path' : '',
                          'additional_general_path' :'',
                          'trips_path': '',
                          'sumocfg_path' : '',
                    }
        case _:
            # Find available networks in the paths.ini file and show the options available to the user
            options = []
            for option in config['Network Path']: options.append(option)
            raise Exception(f"Network not Found. Following networks are available {options}")

    # If the user has a different file than the default options, append that path.
    #TODO - Handles weight paths correctly
    for file_type in user_dict:
        if file_type in paths_dict:
            paths_dict[file_type] = user_dict[file_type]
        else: raise Exception(f'{file_type} from the user, does not match the required path input. Available paths {paths_dict.keys()}')

    # Check if any value is empty, if it is report the error
    #TODO - Ensure that the values are a string.
    #TODO - Ensure that weights are correct
    #TODO - Ensure noone can change sumo path
    if all(value == '' for value in paths_dict.values()): raise Exception('A few paths are empty, please ensure that you provide the data correctly.')
    return paths_dict

#SECTION - Weight files

def generate_weights_files(weight_data, t_start, t_end, type, file_path):

    # Define the root element
    root = ET.Element("edgedata")

    # Create an interval element
    interval = ET.SubElement(root, "interval", id=type, begin=str(t_start), end=str(t_end))

    # Loop over each edge
    for edge, weight in weight_data:
        edge_str = str(edge)
        #TODO - Check if edge is in the network or not. If not throw a warning.
        # Create an edge element
        edge_elem = ET.SubElement(interval, "edge", id=edge_str, value=str(weight))

    # Prettify the XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml()

    #TODO - Make an easy debugger

    # Store the data with the sumo naming convention for a weights file
    with open(f"{file_path}.{type}.xml", "w") as f:
        f.write(pretty_xml)

#SECTION - Demand Generation
def generate_demand_configuration(network:str, usr_config):

    # Gnerate a random seed
    random_seed = random.randint(0,1000000)

    # Get the default demand configuration for the network
    match network:

        case "iisc":
            default_configuration = {
                    'Fringe_Factor':'6',
                    'Fringe_minimum_length':'6000.0',
                    'Travel_minimum_distance':'6000.0',
                    'Minimum_distance_fringe':'10.0',
                    'Random_seed':random_seed,
                    'Max_Tries':'500',
                    'intermediate' : '1'
                    }
            
        case "cubbon":
            default_configuration = {
                    'Fringe_Factor':'4',
                    'Fringe_minimum_length':'500.0',
                    'Travel_minimum_distance':'150.0',
                    'Minimum_distance_fringe':'5.0',
                    'Random_seed':random_seed,
                    'Max_Tries':'500',
                    'intermediate' : '1'
                    }
            
        case "my network":
            default_configuration = {
                    'Fringe_Factor':'4',
                    'Fringe_minimum_length':'500.0',
                    'Travel_minimum_distance':'150.0',
                    'Minimum_distance_fringe':'5.0',
                    'Random_seed':'42',
                    'Max_Tries':'500',
                    'intermediate' : '1'
                    }
        case _:
            raise Exception("Error: case is mishandeled at generate_demand_configuration.")
    
    # If the user has a different values, than the default options, change those values.
    for config_data in usr_config:
        if config_data in default_configuration:
            default_configuration[config_data] = str(usr_config[config_data])
        else: print(f'Ignoring {config_data} from the user, as it does not match any available config Data. Available config Data {default_configuration.keys()}')

    for value in default_configuration.values():
        if value == '': raise Exception('A few config Data, are empty. Please ensure that you provide the data correctly.')
    
    return default_configuration

def generate_demand(paths_dict, weights_path, configuration_dict, veh_per_hour, t_start, t_end): 

    # Generate a temporary file to hold trips and routes data
    temp_trips_file = tempfile.NamedTemporaryFile(dir=paths_dict['demand_output_path'], suffix='.trips.xml', delete=False)
    temp_routes_file = tempfile.NamedTemporaryFile(dir=paths_dict['demand_output_path'], suffix='.rou.xml', delete=False)
    
    # Stop accessing the temprorary files
    temp_trips_file.close()
    temp_routes_file.close()

    #TODO - hold this data seperately in the simulation_generator.py
    #Extract vtypes from additional file
    vtype_tree = ET.parse(paths_dict['additional_general_path'])
    root = vtype_tree.getroot()
    vtype_elements = root.findall('vType')
    vtype_ids = [elem.get('id') for elem in vtype_elements]
    vtype_data = ', '.join(vtype_ids)
    print(vtype_data)


    # Notify the user about where demand progress is made
    #TODO - Allow multiple addtional files to be read
    #TODO - Give relative path to additional file
    #TODO - Fix additional file conflicts
    print(f'Generating Random trips from {t_start} to {t_end}:')
    
    randomTrips.main(randomTrips.get_options([
    '--net-file', paths_dict['network_path'],
    '--output-trip-file',temp_trips_file.name,
    '--route-file',temp_routes_file.name,
    #'--additional-files',paths_dict['additional_general_path'],
    #'--vehicle-class', vtype_data,
    '--weights-prefix', weights_path,
    '--random',
    '--lanes',
    '--remove-loops',
    '--validate',
    '--seed',configuration_dict['Random_seed'],
    '--maxtries',configuration_dict['Max_Tries'],
    '--fringe-factor',configuration_dict['Fringe_Factor'],
    '--allow-fringe.min-length',configuration_dict['Fringe_minimum_length'],
    '--min-distance',configuration_dict['Travel_minimum_distance'],
    '--min-distance.fringe',configuration_dict['Minimum_distance_fringe'],
    '--intermediate',configuration_dict['intermediate'],
    '--insertion-rate',str(veh_per_hour),
    '--begin',str(t_start),
    '--end',str(t_end),
    ]))
    
    return (temp_trips_file, temp_routes_file)

def merge_trips(trips_path, temp_trips_file, temp_routes_file):
    print(trips_path,temp_trips_file.name)
    # Create a new XML tree
    root = ET.Element("routes", attrib={'xmlns:xsi':'http://www.w3.org/2001/XMLSchema-instance', 'xsi:noNamespaceSchemaLocation':'http://sumo.dlr.de/xsd/routes_file.xsd'})

    # List of trips XML files to merge
    files=[trips_path, temp_trips_file.name]
    
    #TODO - Ensure that you append data rather than rereading all the data multiple times and recreating an xml file
    #TODO - Use time as a factor to sort the data. Prehaps not, since we expect files to come in order
    # Counter to ensure uniqness of trip data
    id_counter = 0

    for file in files:
        # Parse an XML file
        try: tree = ET.parse(file)
        except FileNotFoundError: continue

        # Get the root of the parsed XML file
        root_file = tree.getroot()

        # Iterate over each <trip> element in the parsed XML file
        for trip in root_file.findall('trip'):
            # Modify the trip's id to ensure uniqueness
            trip.set('id', str(id_counter))

            # Add the trip to the new XML tree
            root.append(trip)

            # Increment the ID counter
            id_counter += 1

    # Write the new XML tree to a file
    tree = ET.ElementTree(root)
    tree.write(trips_path, xml_declaration=True, encoding='utf-8', method="xml")
    print('Successfully created Trips file')
    
    delete_files(temp_trips_file.name)
    delete_files(temp_routes_file.name)

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

#SECTION - Sumocfg Generation
#TODO - Allow user to change default values.
#TODO - Add additional files data
def generate_sumocfg(network_path, trips_path, config_path):

    # Get the default sumoconfig configuration for the network
    sumoConfig_default_configuration =  {
        "time_to_teleport": '3600',
        "ignore_route_errors": 'True',
        "tls_actuated_jam_threshold": '30',
        "device.rerouting.probability":'1',
    }

    sumoConfig_configuration = sumoConfig_default_configuration
    # Create the root XML element
    root = ET.Element("configuration")

    # Create an input element along with its subelements
    input = ET.SubElement(root, "input")
    ET.SubElement(input, "net-file", value=network_path)
    ET.SubElement(input, "route-files", value=trips_path)
    #ET.SubElement(input, "additional-files", value=additional_files_path)

    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "time-to-teleport", value=sumoConfig_configuration['time_to_teleport'])
    ET.SubElement(processing, "ignore-route-errors", value=sumoConfig_configuration['ignore_route_errors'])

    routing = ET.SubElement(root, "routing")
    ET.SubElement(routing, "device.rerouting.probability", value=sumoConfig_configuration['device.rerouting.probability'])

    # Prettify the XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml()

    # Store the data with the sumo naming convention for a weights file 
    with open(f"{config_path}", "w") as f:
        f.write(pretty_xml)
    
    print(f'Successfully created {config_path}')

