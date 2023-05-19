import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import scipy.linalg
from _fcn.HDV_dynamics import HDV_dynamics
from _fcn.measure_mixed_traffic import measure_mixed_traffic
from _fcn.hankel_matrix import hankel_matrix
from _fcn.qp_DeeP_LCC import qp_DeeP_LCC
from _fcn.qp_MPC import qp_MPC
import math
from ttictoc import tic,toc


'''
                               NEDC Simulation
 Scenario:
       The head vehicle follows a trajectory modified from the Extra-Urban
       Driving Cycle (EUDC) in New European Driving Circle (NEDC)

 See Section V of the following paper for details
   Title : Data-Driven Predicted Control for Connected and Autonomous
           Vehicles in Mixed Traffic
   Author: Jiawei Wang, Yang Zheng, Qing Xu and Keqiang Li
'''

# Parameter setup

# Scenario Setup
# whether traffic flow is mixed

# Set trajectory_id = "1"   test
#trajectory_id = "1"

mix                 = 1                    # 0. all HDVs; 1. there exist CAVs
ID                  = [0,0,1,0,0,1,0,0]    # ID of vehicle types
                                            # 1: CAV  0: HDV
pos_cav             = np.argwhere(np.ravel(ID) == 1)          # position of CAVs
n_vehicle           = len(ID)           # number of vehicles
n_cav               = len(pos_cav)      # number of CAVs
n_hdv               = n_vehicle-n_cav      # number of HDVs

# Definition for Head vehicle trajectory
head_vehicle_trajectory_tmp        = loadmat('./_data/nedc_modified_v1.mat')
head_vehicle_trajectory        = {"time": head_vehicle_trajectory_tmp["time"][0], "vel": head_vehicle_trajectory_tmp["vel"]}
end_time                       = head_vehicle_trajectory["time"][-1]    # end time for the head vehicle trajectory
head_vehicle_trajectory["vel"] = head_vehicle_trajectory["vel"]/3.6

# Initialization Time
initialization_time         = 30               # Time for the original HDV-all system to stabilize
adaption_time               = 20               # Time for the CAVs to adjust to their desired state
# Total Simulation Time
total_time                  = initialization_time + adaption_time + end_time
Tstep                       = 0.05             # Time Step
total_time_step             = int(total_time//Tstep)

# HDV setup

# Type for HDV car-following model
hdv_type            = 1    # 1. OVM   2. IDM
# Parameter setup for HDV
data_str            = '2'  # 1. random ovm  2. manual heterogeneous ovm  3. homogeneous ovm
if hdv_type == 1:
    hdv_parameter_tmp = loadmat('_data/hdv_ovm_'+str(data_str)+'.mat')
    hdv_parameter = {"type": hdv_parameter_tmp["hdv_parameter"]["type"][0][0][0][0],
                     "alpha": hdv_parameter_tmp["hdv_parameter"]["alpha"][0][0],
                     "beta": hdv_parameter_tmp["hdv_parameter"]["beta"][0][0],
                     "s_st": hdv_parameter_tmp["hdv_parameter"]["s_st"][0][0][0][0],
                     "s_go": hdv_parameter_tmp["hdv_parameter"]["s_go"][0][0],
                     "v_max": hdv_parameter_tmp["hdv_parameter"]["v_max"][0][0][0][0],
                     "s_star": hdv_parameter_tmp["hdv_parameter"]["s_star"][0][0]}

elif hdv_type == 2:
    hdv_parameter_tmp = loadmat('_data/hdv_idm.mat')
    hdv_parameter = {"type": hdv_parameter_tmp["hdv_parameter"]["type"][0][0][0][0],
                     "v_max": hdv_parameter_tmp["hdv_parameter"]["v_max"][0][0][0][0],
                     "T_gap": hdv_parameter_tmp["hdv_parameter"]["T_gap"][0][0],
                     "a": hdv_parameter_tmp["hdv_parameter"]["a"][0][0][0][0],
                     "b": hdv_parameter_tmp["hdv_parameter"]["b"][0][0][0][0],
                     "delta": hdv_parameter_tmp["hdv_parameter"]["delta"][0][0][0][0],
                     "s_st": hdv_parameter_tmp["hdv_parameter"]["s_st"][0][0][0][0],
                     "s_star": hdv_parameter_tmp["hdv_parameter"]["s_star"][0][0]}

# Uncertainty for HDV acceleration
acel_noise          = 0.1  # A white noise signal on HDV's acceleration

'''
       Formulation for DeeP-LCC
'''

# Parameter setup

# Type of the controller
controller_type     = 1    # 1. DeeP-LCC  2. MPC
# Initialize Equilibrium Setup (they might be updated in the control process)
v_star              = 15   # Equilibrium velocity
s_star              = 20   # Equilibrium spacing for CAV
# Horizon setup
Tini                = 20   # length of past data in control process
N                   = 50   # length of future data in control process
# Performance cost
weight_v            = 1    # weight coefficient for velocity error
weight_s            = 0.5  # weight coefficient for spacing error
weight_u            = 0.1  # weight coefficient for control input
# Setup in DeeP-LCC
T                   = 2000 # length of data samples
lambda_g            = 100  # penalty on ||g||_2^2 in objective
lambda_y            = 1e4  # penalty on ||sigma_y||_2^2 in objective
# Constraints
constraint_bool     = 1    # whether there exist constraints
acel_max            = 2    # maximum acceleration
dcel_max            = -5   # minimum acceleration (maximum deceleration)
spacing_max         = 40   # maximum spacing
spacing_min         = 5    # minimum spacing
u_limit             = [dcel_max, acel_max]
s_limit             = [spacing_min-s_star, spacing_max-s_star]
# what signals are measurable (for output definition)
measure_type        = 3    # 1. Only the velocity errors of all the vehicles are measurable;
                            # 2. All the states, including velocity error and spacing error are measurable;
                            # 3. Velocity error and spacing error of the CAVs are measurable,
                            #    and the velocity error of the HDVs are measurable.

# Process parameters
n_ctr           = 2 * n_vehicle    # number of state variables
m_ctr           = n_cav          # number of input variables
if measure_type == 1:
    p_ctr = n_vehicle  # number of output variables
elif measure_type == 2:
    p_ctr = 2 * n_vehicle
elif measure_type == 3:
    p_ctr = n_vehicle + n_cav

Q_v         = weight_v*np.eye(n_vehicle)          # penalty for velocity error
Q_s         = weight_s*np.eye(p_ctr-n_vehicle)    # penalty for spacing error
Q           = scipy.linalg.block_diag(Q_v,Q_s)               # penalty for trajectory error
R           = weight_u*np.eye(m_ctr)              # penalty for control input
print(m_ctr)
print(total_time_step)
u           = np.zeros((m_ctr, total_time_step))     # control input
x           = np.zeros((n_ctr, total_time_step))     # state variables
y           = np.zeros((p_ctr, total_time_step))     # output variables
pr_status   = np.zeros((total_time_step, 1))         # problem status
e           = np.zeros((1, total_time_step))         # external input

# Pre-collected data

# load pre-collected data for DeeP-LCC
i_data              = 1    # id of the pre-collected data
pre_data_tmp = loadmat('./data_generation/trajectory_data_collection/data'+data_str+'_'+str(i_data)+'_noiseLevel_'+str(acel_noise)+'.mat')
pre_data = {"Ef": pre_data_tmp["Ef"],
            "Ep": pre_data_tmp["Ep"],
            "hdv_type": pre_data_tmp["hdv_type"][0][0],
            "acel_noise": pre_data_tmp["acel_noise"][0][0],
            "N": pre_data_tmp["N"][0][0],
            "T": pre_data_tmp["T"][0][0],
            "Tini": pre_data_tmp["Tini"][0][0],
            "Tstep": pre_data_tmp["Tstep"][0][0],
            "v_star": pre_data_tmp["v_star"][0][0],
            "Uf": pre_data_tmp["Uf"],
            "Up": pre_data_tmp["Up"],
            "Yf": pre_data_tmp["Yf"],
            "Yp": pre_data_tmp["Yp"],
            "ID": pre_data_tmp["ID"]}

# Simulation

# Mixed traffic states
# S(time,vehicle id,state variable), in state variable: 1. position; 2. velocity; 3. acceleration
S                 = np.zeros((total_time_step,n_vehicle+1,3))
S[0, 0, 0]        = 0
for i in range(1, n_vehicle):
    S[0, i, 0]    = S[0, i-1, 0] - hdv_parameter["s_star"][i-1]
S[0,:,1]          = (v_star * np.ones((n_vehicle+1,1))).T

# reference trajectory is all zeros: stabilize the system to equilibrium
r                 = np.zeros((p_ctr,total_time_step+N))

# Experiment starts here
tic()
# Initialization: all the vehicles use the HDV model
for k in range(int(initialization_time/Tstep)):
    # calculate acceleration
    acel = HDV_dynamics(S[k,:,:], hdv_parameter)-acel_noise + 2 * acel_noise * np.random.rand(n_vehicle, 1)
    S[k, 0, 2]  = 0
    S[k, 1:, 2] = acel.reshape((8,))

    # update traffic states
    S[k + 1,:, 1]      = S[k,:, 1] + Tstep * S[k,:, 2]
    S[k + 1, 0, 1] = head_vehicle_trajectory["vel"][0]
    S[k + 1,:, 0]      = S[k,:, 0] + Tstep * S[k,:, 1]

    # update equilibrium velocity
    v_star = head_vehicle_trajectory["vel"][0]
    # update states in DeeP-LCC
    y[:, k]          = measure_mixed_traffic(S[k, 1:, 1], S[k,:, 0], ID, v_star, s_star, measure_type).reshape((10,))
    #print(e)
    e[0][k]             = S[k, 0, 1] - v_star
    u[:, k]          = S[k, pos_cav, 2].reshape((2,))

# update past data in control process
uini                = u[:, k-Tini:k]            # k???
yini                = y[:, k-Tini:k]
eini                = S[k-Tini:k,0,1] - v_star


# The CAVs start to use DeeP-LCC
for k in range(int(initialization_time/Tstep), total_time_step-1):
    # calculate acceleration for the HDVs
    acel = HDV_dynamics(S[k,:,:], hdv_parameter)-acel_noise + 2 * acel_noise * np.random.rand(n_vehicle, 1)
    S[k, 1:, 2]    = acel.reshape((8,))

    if mix:
        if controller_type == 1:
            if constraint_bool:
                [u_opt, y_opt, pr] = qp_DeeP_LCC(pre_data["Up"], pre_data["Yp"], pre_data["Uf"], pre_data["Yf"], pre_data["Ep"], pre_data["Ef"], uini, yini, eini, Q, R, r[:, k: k + N], lambda_g, lambda_y, u_limit, s_limit)
            else:
                [u_opt, y_opt, pr] = qp_DeeP_LCC(pre_data["Up"], pre_data["Yp"], pre_data["Uf"], pre_data["Yf"], pre_data["Ep"], pre_data["Ef"], uini, yini, eini, Q, R, r[:, k: k + N], lambda_g, lambda_y)
        elif controller_type == 2:
            if constraint_bool:
                [u_opt, y_opt, pr] = qp_MPC(ID, Tstep, hdv_type, measure_type, v_star, uini, yini, N, Q, R,
                                            r[:, k: k + N], u_limit, s_limit)
            else:
                [u_opt, y_opt, pr] = qp_MPC(ID, Tstep, hdv_type, measure_type, v_star, uini, yini, N, Q, R,
                                            r[:, k: k + N])
        # one-step implementation in receding horizon manner
        u[:, k]                      = u_opt[0: m_ctr, 0]
        # update accleration for the CAV
        S[k, pos_cav, 2] = u[:, k].reshape((2,1))
        # judge whether AEB (automatic emergency braking, which is implemented in the function of 'HDV_dynamics') commands to brake
        brake_vehicle_ID = np.argwhere(np.ravel(acel) == dcel_max)  # the vehicles that need to brake
        brake_cav_ID = np.intersect1d(brake_vehicle_ID, pos_cav)  # ,return_indices=True # the CAVs that need to brake

        if not np.all(brake_cav_ID==0):
            S[k, brake_cav_ID, 2] = dcel_max
        # record problem status
        pr_status[k] = pr


    # update traffic states
    S[k + 1,:, 1]      = S[k,:, 1] + Tstep * S[k,:, 2]
    # trajectory for the head vehicle
    # before adaption_time, the head vehicle maintains its velocity and the CAVs first stabilize the traffic system
    if k * Tstep < initialization_time + adaption_time:
        S[k + 1, 0, 1] = head_vehicle_trajectory["vel"][1]
    else:
        print(initialization_time + adaption_time)
        S[k + 1, 0, 1] = head_vehicle_trajectory["vel"][int(k - (initialization_time + adaption_time) / Tstep + 1)]
    S[k + 1,:, 0]      = S[k,:, 0] + Tstep * S[k,:, 1]

    # update equilibrium setup for the CAVs
    v_star = np.mean(S[k - Tini + 1:k+1, 0, 1])    # average velocity of the head vehicle among the past Tini time
    s_star = math.acos(1 - v_star / 30 * 2) / math.pi * (35 - 5) + 5  # use the OVM-type spacing policy to calculate the equilibrium spacing of the CAVs

    # update past data in control process
    uini = u[:, k - Tini + 1: k+1]
    # the output needs to be re-calculated since the equilibrium has been updated
    for k_past in range(k-Tini+1, k+1):
        y[:, k_past] = measure_mixed_traffic(S[k_past, 1:, 1], S[k_past,:, 0], ID, v_star, s_star, measure_type).reshape((10,))
        e[0][k_past] = S[k_past, 0, 1] - v_star
    yini = y[:, k - Tini + 1: k+1]
    eini = S[k - Tini + 1:k+1, 0, 1] - v_star

    # 改一下%号！
    print('Current simulation time: %.2f seconds (%.2f%%) \n',k*Tstep,(k*Tstep-initialization_time)/(total_time-initialization_time)*100.0)

k_end = k+1
y[:,k_end] = measure_mixed_traffic(S[k_end,1:,1],S[k_end,:,0],ID,v_star,s_star,measure_type)

toc()

# Results output
if mix:
    if controller_type == 1:
        save_data = {"hdv_type": hdv_type, "acel_noise": acel_noise, "S": S, "T": T, "Tini": Tini, "N": N, "ID": ID, "Tstep": Tstep, "v_star": v_star}
        save_name = '../_data/simulation_data/DeeP_LCC/nedc_simulation/simulation_data'+ data_str +'_'+str(i_data)+'_modified_v'+str(trajectory_id)+'_noiseLevel_'+str(acel_noise)+'_hdvType_'+str(hdv_type)+'_lambdaG_'+str(lambda_g)+'_lambdaY_'+str(lambda_y)+'.mat'
        savemat(save_name, save_data)
    elif controller_type == 2:
        save_data = {"hdv_type": hdv_type, "acel_noise": acel_noise, "S": S, "T": T, "Tini": Tini, "N": N, "ID": ID,
                     "Tstep": Tstep, "v_star": v_star}
        save_name = '../_data/simulation_data/MPC/nedc_simulation/simulation_data' + data_str + '_' + str(
            i_data) + '_modified_v' + str(trajectory_id) + '_noiseLevel_' + str(acel_noise) + '_hdvType_' + str(
            hdv_type) + '.mat'
        savemat(save_name, save_data)

else:
    save_data = {"hdv_type": hdv_type, "acel_noise": acel_noise, "S": S, "T": T, "Tini": Tini, "N": N, "ID": ID,
                 "Tstep": Tstep, "v_star": v_star}
    save_name = '../_data/simulation_data\HDV/nedc_simulation/simulation_data' + data_str + '_' + str(
        i_data) + '_modified_v' + str(trajectory_id) + '_noiseLevel_' + str(acel_noise) + '_hdvType_' + str(
        hdv_type) + '.mat'
    savemat(save_name, save_data)

