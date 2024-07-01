import xml.etree.ElementTree as ET
from xml.dom import minidom

def fetch_stop_edges(edgeId, tree, root):

    # Find all 'connection' elements in the XML
    connections = root.findall('.//connection')

    # Prepare a list to store the 'from' attributes
    from_list = []

    # Iterate over the 'connection' elements
    for connection in connections:
        # Check if the 'to' attribute matches the provided edgeId
        if connection.get('to') == edgeId:
            # If it does, add the 'from' attribute to the list
            from_list.append(connection.get('from'))

    # Return the list
    return from_list

def check_if_edge_exists(edgeID, tree, root):
    # Parse the XML file

    # Find the edge with the given ID
    edge = root.find(f".//edge[@id='{edgeID}']")

    if edge is None:
        return False
    else: return True

def remove_internal_edges(listOfEdges, tree, root):
    # Parse the XML file
    listOfExternalEdges = []

    for edgeID in listOfEdges:
        # Find the edge with the given ID
        edge = root.find(f".//edge[@id='{edgeID}']")

        if edge is not None:
            # Check if the edge is internal
            function = edge.get('function')
            if function == 'internal':
                continue
            else:
                listOfExternalEdges.append(edgeID)
    
    return listOfExternalEdges

def find_edge_id_for_lane( lane_id, tree, root):

    # Find the lane element with the given ID
    lane = root.find(".//lane[@id='{}']".format(lane_id))
    print(f'laneID = {lane_id}, lane = {lane}')

    if lane is None:
        return None

    edge = lane.find('..')
    print(f'edge = {edge}')

    # Return the id attribute of the edge element
    return edge.get("id")

# Fetch Network Tree and Root Details
network_tree = ET.parse('Input_File\IISC\IISC.net.xml')
network_root = network_tree.getroot()
print(find_edge_id_for_lane('378468045_1', network_root))

def streetClose(edgeDict,  network_root, Output_File = "streetClose.add.xml"):
    # Create the root element
    root = ET.Element("additional")

    # Iterate through the edgeDict
    #TODO - Ensure unique edgeID
    for edge, data in edgeDict.items():
        edgeID, probability, allow = edge
        intervals = data

        # Check if edge exists in the map
        if not check_if_edge_exists(edgeID,  network_root):
            print(f"Ignoring Street Blocking for {edgeID}, as it does not exist in the network file")
            continue

        # Fetch the approach edges using the fetch_stop_edges function
        approach_edges = fetch_stop_edges(edgeID,  network_root)

        # Filter the approach edges to obtain the stop edges using the remove_internal_edges function
        stop_edges = remove_internal_edges(approach_edges,  network_root)

        # Create a rerouter element
        rerouter = ET.SubElement(root, "rerouter", id=f"REROUTER_{edgeID}", edges=f"{edgeID} {' '.join(stop_edges)}", probability=str(probability))

        # Iterate through the intervals
        for begin, end in intervals:
            # Create an interval element
            interval = ET.SubElement(rerouter, "interval", begin=begin, end=end)

            # Create a closingReroute element
            closingReroute = ET.SubElement(interval, "closingReroute", id=edgeID, allow=allow)

    # Prettify the XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml()

    # Save the XML to a file
    with open(Output_File, "w") as f:
        f.write(pretty_xml)

def laneClose(laneDict,  network_root, Output_File = "laneClose.add.xml"):

    # Create the root element
    root = ET.Element("additional")

    # Iterate through the laneDict
    for lane, data in laneDict.items():
        laneID, probability, allow = lane
        intervals = data

        # Fetch edgeID for the given laneID
        edgeID = find_edge_id_for_lane(laneID,  network_root)
        if edgeID == False: continue

        # Fetch the approach edges using the fetch_stop_edges function
        approach_edges = fetch_stop_edges(edgeID,  network_root)

        # Filter the approach edges to obtain the stop edges using the remove_internal_edges function
        stop_edges = remove_internal_edges(approach_edges,  network_root)

        # Create a rerouter element
        rerouter = ET.SubElement(root, "rerouter", id=f"REROUTER_{laneID}", edges=f"{edgeID} {' '.join(stop_edges)}", probability=str(probability))

        # Iterate through the intervals
        for begin, end in intervals:
            # Create an interval element
            interval = ET.SubElement(rerouter, "interval", begin=begin, end=end)

            # Create a closingReroute element
            closingReroute = ET.SubElement(interval, "closingLaneReroute", id=laneID, allow=allow)

    # Prettify the XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml()

    # Save the XML to a file
    with open(Output_File, "w") as f:
        f.write(pretty_xml)


edgeDict = {
    ("249906101", 1.0, ""): [("9:0:0", "9:30:0"), ("0:40:0", "9:30:0")],
    ("378425665", 1.0, ""): [("1:0:0", "3:30:0"), ("3:40:0", "13:30:0")],
    ("323930524#0", 1.0, ""): [("1:0:0", "3:30:0"), ("3:40:0", "13:30:0")],
}

laneDict = {
    ("383031432#9_0", 1.0, ""): [("1:0:0", "3:30:0"), ("3:40:0", "13:30:0")],
}

# Fetch Network Tree and Root Details
network_tree = ET.parse('Input_File\IISC\IISC.net.xml')
network_root = network_tree.getroot()

# print(get_edge_from_lane('random.net.xml','-1550_0'))
streetClose(edgeDict, network_root)
# laneClose(laneDict,  network_root)