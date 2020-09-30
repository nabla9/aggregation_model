/* This file initializes the proper database structure for storing swarming simulation data. */ 

CREATE DATABASE swarmsim;

USE swarmsim;

/* Parent table, contains unique identifier for simulation and most input parameters. */
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

/* Child to "simulations," contains community data. */
CREATE TABLE communities (
	sim_id int NOT NULL 
	, comm_id int NOT NULL
	, comm_nodes int NOT NULL
	, UNIQUE (sim_id,comm_id)
	, FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
);

/* Child to "simulations," used to initialize simulation run plan. */
CREATE TABLE runs (
    sim_id int NOT NULL
    , n_runs int NOT NULL
    , p_inner float(2) NOT NULL
    , p_outer float(2) NOT NULL
    , done datetime NULL
    , UNIQUE (sim_id, p_inner, p_outer)
    , FOREIGN KEY (sim_id) REFERENCES simulations(sim_id)
);

/* Child to "simulations," contains large amount of records (for each time step of each simulation). */
CREATE TABLE statedata (
	sim_id int NOT NULL
	, run_id int NOT NULL
	, node int NOT NULL
	, step_time float(2) NOT NULL 
	, node_pos float(2) NOT NULL
	, UNIQUE (sim_id,run_id,node,step_time)
	, FOREIGN KEY (sim_id) REFERENCES simulations(sim_id) 
);