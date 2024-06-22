# Road Traffic Analytics using GNN and SUMO

This project aims to analyze road traffic using Graph Neural Networks (GNN) and the Simulation of Urban MObility (SUMO) tool. The project structure and directory organization are described below.

## Directory Structure

### SUMO

This directory holds data obtained from SUMO simulations.

#### IISC

This subdirectory contains map data for region A.

- **Network**: Holds the map data.
- **Common_Additional_file**: Contains data used by all simulations except for the TAZ simulation.
- **Plain Network**: Directory for demand data of the network.
- **TAZ**: Directory for TAZ (Traffic Analysis Zone) data.
  - **Additional_file_Data**: Data used instead of the common additional file.
  - **Demand Data**: Demand data specific to the TAZ network.
  - **OD Matrix**: Origin-Destination Matrix for TAZ.
- **Virtual Node**: Contains unique network files for virtual nodes and related demand files.
  - **Weights**: Directory holding `src`, `dst`, and `via` files.

#### Cubbon

- **Network**: Holds the map data.
- **Common_Additional_file**: Contains data used by all simulations except for the TAZ simulation.
- **Plain Network**: Directory for demand data of the network.
- **TAZ**: Directory for TAZ (Traffic Analysis Zone) data.
  - **Additional_file_Data**: Data used instead of the common additional file.
  - **Demand Data**: Demand data specific to the TAZ network.
  - **OD Matrix**: Origin-Destination Matrix for TAZ.
- **Virtual Node**: Contains unique network files for virtual nodes and related demand files.
  - **Weights**: Directory holding `src`, `dst`, and `via` files.
