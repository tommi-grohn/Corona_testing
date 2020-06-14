import requests
import pandas as pd
import arviz as az
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import date, timedelta
import datetime
from scipy.integrate import solve_ivp
from scipy import integrate, optimize
import numpy as np

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az
import theano

##########################################################################

train_model = True  # Set true if you want to train the model

##########################################################################

# Timeperiods defined

len_one_estimation_period = 7
estimation_periods = 6   # There is manual part in the SEIR function that needs to be changed if this too..
len_estimation_period = estimation_periods * len_one_estimation_period

len_prediction_period = 14

infected_at_first = 10   # Before at least this amount of people are infected, ignore data

# Constant parameters of the model
D_e = 3
D_i = 2
D_r = 5

alpha_1 = 1.25
alpha_2 = 0.1

r = 0.4

time_shift = 7   # the time it takes newly infected 


# In general, the population is chosen based on metropolian areas

###############################
#city_name = "Berlin"
#county_name = "Berlin"
#population = 3770000

###############################
city_name = "Stockholm"
county_name = "Stockholm County"
population = 975000


##########################################################################

# Read data and make sure for each day there is new daily cases, also for missing data
# REMARK: Maybe some kind of convolution also for case data would be nice

df = pd.read_csv('regional_data.csv')
df = df.loc[(df['City'] == city_name)]

dates = df['Date'].tolist()
for ind in list(range(len(dates))): 
    dates[ind] = datetime.datetime.strptime(dates[ind],"%Y-%m-%d") - timedelta(days=time_shift)

ts = pd.Series(df['New cases'].tolist(), dates)
ts = ts.resample('D').mean()
ts = ts.interpolate()
ts = ts.astype('int64')

ts = ts.rolling(window = 7).mean()

# Create timelines, newly observed cases and cumulative cases based on the value infected_at_first

one_estimation_period = np.arange(len_one_estimation_period)
estimation_period = np.arange(len_estimation_period)
total_time_period = np.arange(len_estimation_period + len_prediction_period)

all_cases = df['New cases'].to_numpy()
obs_new = np.zeros(len_estimation_period + len_prediction_period)

start_index = 0
while all_cases[start_index] < infected_at_first:
    start_index += 1

running_index = start_index
for i in range(len_estimation_period + len_prediction_period):
    obs_new[i] = all_cases[running_index]
    running_index = running_index + 1

start_date = df['Date'].to_numpy()[start_index]
mid_date = df['Date'].to_numpy()[start_index + len_estimation_period]
end_date = df['Date'].to_numpy()[start_index + len_estimation_period + len_prediction_period]

# obs_new is a list of daily new cases. This will not be used in the estimation!
obs_cum = obs_new.copy()
ind = 1
while ind <= len(obs_new) - 1:
    obs_cum[ind] = obs_cum[ind] + obs_cum[ind-1]
    ind = ind + 1


##########################################################################

N = population 

# Initial stages
exp0 = obs_cum[6] - obs_cum[1]
inf0 = obs_cum[0]
asy0 = obs_cum[0]
sus0 = N - exp0 - inf0 -asy0
hos0 = 0

# s: y[0], e: y[1], i_1: y[2], i_2: y[3], a: y[4]
def SEIR(y, t, p):
    ds = -p[0] * y[0] * (y[3] + alpha_1 * y[2] + alpha_2 * y[4]) / N
    de =  p[0] * y[0] * (y[3] + alpha_1 * y[2] + alpha_2 * y[4]) / N - (1-r) * y[1] / D_e - r * y[1] / D_e
    di_1 = (1-r) * y[1] / D_e - y[2] / D_i
    di_2 = y[2] / D_i - y[3] / D_r
    da = r * y[1] / D_e - y[4] / D_r
    
    return [ds, de, di_1, di_2, da]  


seir_model = DifferentialEquation(
    func=SEIR,
    times=one_estimation_period,
    n_states=5, # S, E, I, A, H. (no need to have R)
    n_theta=1,
    t0=0,
)

with pm.Model() as modelSEIR:

    beta1 = pm.Uniform('beta1', 0, 5)
    beta2 = pm.Uniform('beta2', 0, 5)
    beta3 = pm.Uniform('beta3', 0, 5)
    beta4 = pm.Uniform('beta4', 0, 5)
    beta5 = pm.Uniform('beta5', 0, 5)
    beta6 = pm.Uniform('beta6', 0, 5)
    
    seir_curves1 = seir_model(y0 = [sus0, exp0, inf0, asy0, hos0], theta=[beta1,])
    
    seir_curves2 = seir_model(y0 = [seir_curves1[-1 ,0], seir_curves1[-1 ,1], seir_curves1[-1 ,2],
    seir_curves1[-1 ,3], seir_curves1[-1 ,4],], theta=[beta2,])
    
    seir_curves3 = seir_model(y0 = [seir_curves2[-1 ,0], seir_curves2[-1 ,1], seir_curves2[-1 ,2],
    seir_curves2[-1 ,3], seir_curves2[-1 ,4],], theta=[beta3,])
    
    seir_curves4 = seir_model(y0 = [seir_curves3[-1 ,0], seir_curves3[-1 ,1], seir_curves3[-1 ,2],
    seir_curves3[-1 ,3], seir_curves3[-1 ,4],], theta=[beta4,])
    
    seir_curves5 = seir_model(y0 = [seir_curves4[-1 ,0], seir_curves4[-1 ,1], seir_curves4[-1 ,2],
    seir_curves4[-1 ,3], seir_curves4[-1 ,4],], theta=[beta5,])
    
    seir_curves6 = seir_model(y0 = [seir_curves5[-1 ,0], seir_curves5[-1 ,1], seir_curves5[-1 ,2],
    seir_curves5[-1 ,3], seir_curves5[-1 ,4],], theta=[beta6,])
    
    lambda_ = theano.tensor.concatenate(
        [seir_curves1[:,1] * r,
         seir_curves2[:,1] * r,
         seir_curves3[:,1] * r,
         seir_curves4[:,1] * r,
         seir_curves5[:,1] * r,
         seir_curves6[:,1] * r
        ]) / D_e
        
    new_infectious = pm.Poisson('new_infectious', lambda_ ,
                                observed = obs_new[0:len_estimation_period])

if train_model:
    with modelSEIR:
        trace = pm.sample(500,tune=500, target_accept = 0.95, cores=1)
        db = pm.backends.Text('data_trace_stockholm_6_timelines')
        pm.save_trace(trace, directory = 'data_trace_stockholm_6_timelines', overwrite=True)
else:
    with modelSEIR:
        trace = pm.load_trace(directory = 'data_trace_stockholm_6_timelines')

