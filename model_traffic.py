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

##################################################################################
train_model = True  # Set true if you want to train the model

#################################################################################
# Timeperiods defined

len_estimation_period = 50
len_prediction_period = 30

infected_at_first = 10   # Before at least this amount of people are infected, ignore data

# Constant parameters of the model

D_e = 5.2
D_i = 2.3
D_q = 7
D_h = 30

# In general, the population is chosen based on metropolian areas

###############################
city_name = "Berlin"
county_name = "Berlin"
population = 3770000

###############################
#city_name = "Brussels"
#county_name = "Brussels"
#population = 2500000

###############################   # Singapore needs to read data differently... Special case
#city_name = "Singapore"
#population = 5640000

###############################
#city_name = "Stockholm"
#county_name = "Stockholm County"
#population = 975000


#################################################################################

# Read data and make sure for each day there is new daily cases, also for missing data
# REMARK: Maybe some kind of convolution also for case data would be nice

df = pd.read_csv('regional_data.csv')
df = df.loc[(df['City'] == city_name)]

dates = df['Date'].tolist()
for ind in list(range(len(dates))): 
    dates[ind] = datetime.datetime.strptime(dates[ind],"%Y-%m-%d")

ts = pd.Series(df['New cases'].tolist(), dates)
ts = ts.resample('D').mean()
ts = ts.interpolate()
ts = ts.astype('int64')

ts.plot(title="The reported number of new daily Covid-infections")


# Create timelines, newly observed cases and cumulative cases based on the value infected_at_first

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



#################################################################################

def load_mobility_raw_onlynan():
    url = 'Global_Mobility_Report_Germany.csv'
    
    data = pd.read_csv(url, dtype={"country_region_code": "string",
                                   "country_region": "string",
                                   "sub_region_1": "string",
                                   "sub_region_2": "string",
                                   "date": "string",
                                   "retail_and_recreation_percent_change_from_baseline": float,
                                   "grocery_and_pharmacy_percent_change_from_baseline": float,
                                   "parks_percent_change_from_baseline": float,
                                   "transit_stations_percent_change_from_baseline": float,
                                   "workplaces_percent_change_from_baseline": float,
                                   "residential_percent_change_from_baseline": float}
                                   )
    
    data = data[data['sub_region_1'].notna()]
    
    data["date"] = pd.to_datetime(data["date"])
    data['retail_and_recreation_percent_change_from_baseline']=data.groupby('sub_region_1')['retail_and_recreation_percent_change_from_baseline'].fillna(method='ffill')
    data['retail_and_recreation_percent_change_from_baseline']=data.groupby('sub_region_1')['retail_and_recreation_percent_change_from_baseline'].fillna(method='bfill')
    data['grocery_and_pharmacy_percent_change_from_baseline']=data.groupby('sub_region_1')['grocery_and_pharmacy_percent_change_from_baseline'].fillna(method='ffill')
    data['grocery_and_pharmacy_percent_change_from_baseline']=data.groupby('sub_region_1')['grocery_and_pharmacy_percent_change_from_baseline'].fillna(method='bfill')
    data['parks_percent_change_from_baseline']=data.groupby('sub_region_1')['parks_percent_change_from_baseline'].fillna(method='ffill')
    data['parks_percent_change_from_baseline']=data.groupby('sub_region_1')['parks_percent_change_from_baseline'].fillna(method='bfill')
    data['transit_stations_percent_change_from_baseline']=data.groupby('sub_region_1')['transit_stations_percent_change_from_baseline'].fillna(method='ffill')
    data['transit_stations_percent_change_from_baseline']=data.groupby('sub_region_1')['transit_stations_percent_change_from_baseline'].fillna(method='bfill')
    data['workplaces_percent_change_from_baseline']=data.groupby('sub_region_1')['workplaces_percent_change_from_baseline'].fillna(method='ffill')
    data['workplaces_percent_change_from_baseline']=data.groupby('sub_region_1')['workplaces_percent_change_from_baseline'].fillna(method='bfill')
    data['residential_percent_change_from_baseline']=data.groupby('sub_region_1')['residential_percent_change_from_baseline'].fillna(method='ffill')
    data['residential_percent_change_from_baseline']=data.groupby('sub_region_1')['residential_percent_change_from_baseline'].fillna(method='bfill')

    data['workplaces_percent_change_from_baseline']=data.groupby('sub_region_1')['workplaces_percent_change_from_baseline'].apply(lambda group: group.interpolate(method='index'))
    
    datav=data.values
    tot_mobility=list()
    for j in range(datav.shape[0]):
        temp=np.nanmean([datav[j,h] for h in range(5,11,1)])
        tot_mobility.append(temp)

    data['Total Mobility']=tot_mobility
    data['Total Mobility']=data.groupby('sub_region_1')['Total Mobility'].fillna(method='ffill')
    data['Total Mobility']=data.groupby('sub_region_1')['Total Mobility'].fillna(method='bfill')
    data['Smooth Mobility']=data.groupby('sub_region_1')['Total Mobility'].rolling(window = 7).mean().reset_index(0,drop=True)
    data['Smooth Mobility']=data.groupby('sub_region_1')['Smooth Mobility'].fillna(method='ffill')
    data['Smooth Mobility']=data.groupby('sub_region_1')['Smooth Mobility'].fillna(method='bfill')
    data['Smooth Mobility']=data.groupby('sub_region_1')['Smooth Mobility'].apply(lambda group: group.interpolate(method='index'))
    data=data.drop(columns=['country_region_code','country_region', 'sub_region_2'])
    return data

df = load_mobility_raw_onlynan()

# Choose the wanted region
df = df.loc[(df['sub_region_1'] == county_name)]

df = df.loc[(df['date'] >= datetime.datetime.strptime(start_date, '%Y-%m-%d'))
           & (df['date'] <= datetime.datetime.strptime(end_date, '%Y-%m-%d'))]

traffic_data_tp12 = df[0:len_estimation_period+len_prediction_period]["Smooth Mobility"].values
traffic_data_tp12 = np.ones(len_estimation_period+len_prediction_period) + traffic_data_tp12/100

print(traffic_data_tp12)

traffic_data_tp12_tensor = theano.shared(traffic_data_tp12)



#################################################################################
# This part not implemented yet
def current_r(t):
    0.5   # In the first version, let r be a constant

#################################################################################

N = population 

# Initial stages
exp0 = obs_cum[6] - obs_cum[1]
inf0 = obs_cum[0]
asy0 = obs_cum[0]
sus0 = N - exp0 - inf0 -asy0
hos0 = 0


def SEIR(y, t, p):
    t_int = t.astype('int32')
    beta = p[0] * traffic_data_tp12_tensor[t_int] + p[1]
    r = 0.5  # current_r(t_int)
    
    ds = -beta * y[0] * (y[2] + p[2] * y[3]) / N 
    de =  beta * y[0] * (y[2] + p[2] * y[3]) / N - y[1] / D_e
    di =  r * y[1] / D_e - y[2] / D_q - y[2] / D_i
    da =  (1 - r) * y[1] / D_e - y[3] / D_i
    dh =  y[2] / D_q - y[4] / D_h
    
    return [ds, de, di, da, dh]  



seir_model = DifferentialEquation(
    func=SEIR,
    times=estimation_period,  # !!!
    n_states=5, # S, E, I, A, H. (no need to have R)
    n_theta=3,
    t0=0,
)


with pm.Model() as modelSIR:

    c_1 =  pm.Uniform('c_1', 0, 5)
    c_2 =  pm.Uniform('c_2', 0, 5)
    alpha =  pm.Uniform('alpha', 0, 5)

    seir_curves = seir_model(y0 = [sus0, exp0, inf0, asy0, hos0], theta=[c_1, c_2, alpha,])
    lambda_ = seir_curves[: ,1] * 0.5 / D_e   # r = 0.5 for now

    new_infectious = pm.Poisson('new_infectious', lambda_ , observed = obs_new[0:len_estimation_period])


now = datetime.datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


if train_model:
    with modelSIR:
        trace = pm.sample(300,tune=300, target_accept = 0.9, cores=1)
        db = pm.backends.Text('pure_python_test_data')
        pm.save_trace(trace, directory = 'pure_python_test_data', overwrite=True)
else:
    with modelSIR:
        trace = pm.load_trace(directory = 'pure_python_test_data')


now = datetime.datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



pm.traceplot(trace);


az.summary(trace)


az.plot_posterior(trace);


c_1_estimated = np.mean(trace.c_1)
c_2_estimated = np.mean(trace.c_2)
r_estimated = np.mean(trace.r)

