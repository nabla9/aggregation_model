# aggregation_model: a mathematical model of swarming behavior

## What is it? 
**aggregation_model** is a package that provides a flexible numerical implementation of a particular aggregation model as explored in [von Brecht et al.](https://link.springer.com/article/10.1007/s10955-012-0680-x). This is an *individual-based model* that exhibits large-scale pattern formation, as in the below:

![agg-img](placeholder.png)

## Config and installation
### Config file
The main config file of this package is *config.conf* (which should be renamed to *agg.conf* by the user). 

	[SQL]
	host = localhost
	port = 3306
	database = swarmsim
	password = YOURPASS

### Installation
Required packages are in *requirements.txt*; these can be installed by running `pip3 install -r requirements.txt` (Python 3). 

The commands in the provided file *init.sql* should be run once; these schema create the `swarmsim` database and its associated tables: 
1. **simulations**, a parent table that contains fields for initial data from each simulation run;
2. **sim_comms**, a child table that stores community data;
3. **simdata**, another child table that stores the many-dimensional simulation state data.

## Main components
There are several components of this package: 
1. **block_model.py**, a module that offers a function for creating a stochastic block adjacency matrix and a class to properly wrap this matrix to feed into the other modules; 
2. **odetools.py**, a module that provides a function for generating initial conditions with added noise sampled from various random distributions and houses the logic for evolving the system of differential equations;
3. **plottools.py**, a module that wraps the output of *odetools* and supplements it with methods that compute centers of mass from the data, plot state data by community, and call functions to record this data in a SQL database for future processing;
4. **sql_connector.py**, an interface for *mysql-connector-python* that provides methods for easy reading of relevant data and writing of state data to a SQL database.

