#!/usr/bin/env python3
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pulp import LpMaximize, LpMinimize, LpInteger,LpProblem, LpContinuous, LpStatus, lpSum, LpVariable, LpConstraint, LpAffineExpression, LpBinary
import pulp 
import networkx as nx
import time
from mpi4py import MPI

# Travelling Salesperson Problem with Time-Window (MIP).
def tsp_tw(input_df,vehicle_id):
    print(f'Running TSPTW_MIP for vehicle_id: {vehicle_id}')

    #########################----------------------------INITIALIZED----------------------------#########################
    sub_df = input_df.copy()
    item_list = np.arange(len(sub_df))
    sub_df['item_ind'] = item_list
    X = sub_df['location_latitude'].to_numpy()
    Y = sub_df['location_longitude'].to_numpy()
    
    travel_time_matrix = np.sqrt(np.power((X[:,None] - X[None,:]),2) + np.power((Y[:,None] - Y[None,:]),2))
    
    node = sub_df['item_ind'].astype('int').to_numpy()
    model = LpProblem("TSPTW", LpMinimize)


    # Variables
    x = LpVariable.dicts('x',((i,j) for i in node for j in node if i != j), cat='Binary')
    t = LpVariable.dicts('t', ((i) for i in node), lowBound = 0, cat = 'Continuous')

    # Constraints 1: Incoming constraints
    for j in node:
        model += (lpSum(x[(i, j)] for i in node if i != j) == 1)

    # Constraints 2: Outgoing constraints
    for i in node:
        model += (lpSum(x[(i, j)] for j in node if i != j) == 1)

    # Constraints 3: Time Windows
    M = 10000000
    for i in node:
        model += (t[(i)] >= sub_df.loc[sub_df['item_ind'] == i,'ready_time'].values[0])
        model += (t[(i)] <= sub_df.loc[sub_df['item_ind'] == i,'due_time'].values[0])
        
    for i in node:
        for j in node:
            if i > 0 and i != j:
                travel_time = travel_time_matrix[i][j]
                serve_time = sub_df.loc[sub_df['item_ind'] == i,'service_time'].values[0]
                model += (t[(j)] >= t[(i)] + travel_time + serve_time - M*(1 - x[(i,j)]))

            # Start at the depot
            if i == 0 and i != j:
                travel_time = travel_time_matrix[i][j]
                serve_time = sub_df.loc[sub_df['item_ind'] == i,'service_time'].values[0]
                model += (t[(j)] >= 0 + travel_time + serve_time - M*(1 - x[(i,j)]))

    # Objective Function
    cost = lpSum([x[(i,j)]*travel_time_matrix[i][j] for i in node for j in node if i != j])
    model += cost

    #########################----------------------------RUNNING TSPTW_MIP----------------------------#########################
    solver = pulp.COIN_CMD(msg = False,timeLimit=20)

    start_time = time.time()
    status = model.solve(solver)
    stop_time = time.time()
    run_time = stop_time - start_time
    print(f'MIP Running Time for vehicle {vehicle_id}: {run_time}')

    #########################----------------------------SUBTOUR ELIMINATION----------------------------#########################
    routes = [(i,j) for i in node for j in node if i != j if x[i,j].value() == 1]

    while True:
        G = nx.Graph()
        G.add_nodes_from(node)
        for edge in routes:
            G.add_edge(edge[0],edge[1])
        nr = nx.number_connected_components(G)
        if nr > 1:
            # print('Has Subtour')
            components = nx.connected_components(G)
            for c in components:
                model += (lpSum(x[(i,j)] for i in c for j in c if i != j) <= len(c) - 1)
        else:
            # print('No Subtour')
            break
            
    #########################----------------------------GENERATED OUTPUT----------------------------#########################
    df_routes = pd.DataFrame(routes)
    df_routes.columns = ['start','stop']
    output = pd.DataFrame()

    for ind in range(len(df_routes)):
        if len(output) == 0:
            output = df_routes[df_routes['start'] == 0].copy()

            start_node = output['stop'].iloc[0]
        else:
            output = pd.concat([output,df_routes[df_routes['start'] == start_node]],ignore_index=True)
            start_node = output['stop'].iloc[-1]

    output.drop(['stop'],axis = 1,inplace = True)
    output.rename(columns = {'start':'item_ind'},inplace = True)
    output = pd.concat([output,pd.DataFrame({'item_ind':[0]})],ignore_index=True)
    output['vehicle_id'] = vehicle_id
    output = output[['vehicle_id','item_ind']]
    output = output.merge(sub_df[['item_name','item_ind','quantities','ready_time','due_time','service_time']], on = 'item_ind',how = 'left')
    output['item_name'] = output['item_name'].astype('int')

    for ind in range(len(output)):
        if ind == 0:
            vehicle_arrival_time = [0]
            vehicle_departure_time = [0]
            vehicle_travel_dis = [0]
            vehicle_cum_dis = [0]
            start_node = output['item_ind'].iloc[ind]
        else:
            stop_node = output['item_ind'].iloc[ind]
            travel_time = travel_time_matrix[start_node][stop_node]
            cum_travel_dis = vehicle_cum_dis[-1] + travel_time
            vehicle_travel_dis.append(travel_time)
            vehicle_cum_dis.append(cum_travel_dis)

            if  output['ready_time'].iloc[ind] < vehicle_departure_time[-1] + travel_time <= output['due_time'].iloc[ind]:
                arr_time = vehicle_departure_time[-1] + travel_time
                vehicle_arrival_time.append(arr_time)

            elif vehicle_departure_time[-1] + travel_time <= output['ready_time'].iloc[ind]:
                arr_time = output['ready_time'].iloc[ind]
                vehicle_arrival_time.append(arr_time)

            else:
                print('Impossible to arrive in time')

            dep_time = arr_time + output['service_time'].iloc[ind]
            vehicle_departure_time.append(dep_time)

            start_node = stop_node

    output['arrival_time'] = vehicle_arrival_time
    output['departure_time'] = vehicle_departure_time
    output['travel_distance'] = vehicle_travel_dis
    output['cumulative_distance'] = vehicle_cum_dis
    output.drop(['item_ind','ready_time','due_time','service_time'],axis= 1, inplace = True)
    output = output[['vehicle_id', 'item_name', 'arrival_time', 'departure_time','quantities','travel_distance','cumulative_distance']]
    print('\n')
    
    return output,run_time



#############################------------------------------MAIN------------------------------#############################
# Open the communication port for each cores.
comm = 
# Get the rank of each cores.
rank = 
# Get the total cores used.
size = 


if rank == 0: # Master core that runs the main algorithm to clustered the raw dataset first before sending each zones to the slave cores.
    
    # In this case, we assume that the master core has finished clustering algorithm, so just use the provided clustered data.
    df = pd.read_csv('clustered_dataset.csv',index_col=0)
    init_avail_cores = 
    avail_routes = 

    master_output = pd.DataFrame()
    fin_cores = np.array([])
    start_running = time.time()
    
    # Automatically sending each zones/routes to slaves cores, if finished, send signal to the cores and break the loop.
    while True:
        if avail_routes.size == 0:
            # Receving the last sets of outgoing zones.

            
            # Update the output
            master_output = 
            
            # Sending to that core to stop running.

            
            # Store the finished cores.
            fin_cores = 
            print(f'FIN CORES: {fin_cores}')
            
            if fin_cores.size == size - 1:
                break
                

        else:
            # Choose available zones/routes that have not already been running MIP.
            veh_id = 
            avail_routes = 

            # Sub_df
            sub_df = 

            # At FIRST, sending the routes to all cores.
            if init_avail_cores.size != 0:
                # Choose core to send
                core = 
                init_avail_cores = 

                # Send to each core, tagged by core number


            # After sending to all cores, wait for receving the results, and then send new zone/route to that core.
            else:
                # Receive the data

                
                # Update the outputs
                if len(master_output) == 0:
                    master_output = 
                else:
                    master_output = 

                # The chosen core is the same as the core that has just sent the previous result to master.
                core = 

                # Send to that core

                
    stop_running = time.time()
    print(f'Parallel running time: {stop_running - start_running} s')
    master_output = master_output.sort_values(by = ['vehicle_id','cumulative_distance']).reset_index(drop=True)
    print(f'''Total distance = {master_output['travel_distance'].sum()}''')
    print(master_output)

    # Save the output


else: # Slave cores, which will run each clustered zone using TSP_TW MIP
    while True:
        # Receive the data


        if len(data) == 1: # If receive the signal instead of the routes, stop the loop
            break

        else: # If not, running TSP_TW and then send it to the master
            tsp_output, run_time = tsp_tw(input_df=  ,vehicle_id= )

            # Send the output

MPI.Finalize


