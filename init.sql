/* This file initializes the proper database structure for storing swarming simulation data. */ 

CREATE DATABASE swarmsim;

CREATE TABLE simulations (
	sim_id int NOT NULL AUTO_INCREMENT
	, n_nodes int NOT NULL
	, n_comms int NOT NULL
	, graph json NOT NULL
	, ker_a float(2) NOT NULL
	, ker_b float(2) NOT NULL
	, notes varchar(1000)
	, PRIMARY KEY (sim_id)
);

CREATE TABLE simcomms (
	sim_id int NOT NULL 
	, comm_id int NOT NULL
	, comm_nodes int NOT NULL
	, UNIQUE(sim_id,comm_id)
	, FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
);

CREATE TABLE simdata (
	sim_id int NOT NULL
	, node int NOT NULL
	, step_time float(2) NOT NULL 
	, node_pos float(2) NOT NULL
	, UNIQUE (node,step_time,node_pos)
	, PRIMARY KEY (sim_id,node,step_time)
	, FOREIGN KEY (sim_id) REFERENCES simulations(sim_id) 
);