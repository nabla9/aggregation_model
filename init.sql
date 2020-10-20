/* This file initializes the proper database structure for storing swarming simulation data. */ 

CREATE DATABASE swarmsim;

USE swarmsim;

/* Parent table, contains unique identifier for simulation and most input parameters. */
CREATE TABLE simulations (
	sim_id int NOT NULL AUTO_INCREMENT
	, n_runs int
	, n_nodes int NOT NULL
	, n_comms int NOT NULL
	, ker_a float NOT NULL
	, ker_b float NOT NULL
	, notes varchar(1000)
	, PRIMARY KEY (sim_id)
);

/* Child to "simulations," contains community data. */
CREATE TABLE communities (
	sim_id int NOT NULL 
	, comm_id int NOT NULL
	, comm_nodes int NOT NULL
	, UNIQUE (sim_id,comm_id)
	, FOREIGN KEY (sim_id)
	    REFERENCES simulations(sim_id)
	    ON DELETE CASCADE
);

/* Child to "simulations," used to initialize simulation run plan. */
CREATE TABLE runs (
    sim_id int NOT NULL
    , run_id int NOT NULL AUTO_INCREMENT
    , p_inner decimal(3,2) NOT NULL
    , p_outer decimal(3,2) NOT NULL
    , done datetime NULL
    , PRIMARY KEY (run_id)
    , FOREIGN KEY (sim_id)
        REFERENCES simulations(sim_id)
        ON DELETE CASCADE
);

/* Child to "simulations" and "graphs," contains large amount of records (for each time step of each simulation). */
CREATE TABLE statedata (
	sim_id int NOT NULL
	, run_id int NOT NULL
	, node int NOT NULL
	, step_time float NOT NULL
	, node_pos float NOT NULL
	, UNIQUE (sim_id,run_id,node,step_time)
	, FOREIGN KEY (sim_id)
	    REFERENCES simulations(sim_id)
	    ON DELETE CASCADE
	, FOREIGN KEY (run_id)
	    REFERENCES runs(run_id)
	    ON DELETE CASCADE
);

/* Parent to "statedata," stores graph data per simulation run */
CREATE TABLE graphs (
    sim_id int NOT NULL
    , run_id int NOT NULL
    , p_inner decimal(3,2) NOT NULL
    , p_outer decimal(3,2) NOT NULL
    , graph json NOT NULL
    , FOREIGN KEY (sim_id)
        REFERENCES simulations(sim_id)
        ON DELETE CASCADE
    , FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);