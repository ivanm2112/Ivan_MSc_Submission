import pandas as pd
import numpy as np
import os

#step_size=int(os.getenv("step_size"))
step_size = 30
#Loading forecast excel file
data_file = pd.ExcelFile("./CREST_Demand_Model_v2.3.3.xlsm").parse("Results - aggregated")

#Cleaning up the file due to Excel formatting
data_file.columns = data_file.iloc[0]
data_file = data_file.drop([0,1,2])

#Loading individual forecasts
Elec_Demand_Forecast = np.array(data_file["Total electricity demand"])
Therm_forecast = np.array(data_file["Thermal demand for hot water"])
PV_forecast = np.array(data_file["Total PV output"])   #1440 size

#Shortening the arrays to step size defined at the start.
from numpy_utils import shorten_array

Elec_Demand_Forecast = shorten_array(Elec_Demand_Forecast, step_size)
Therm_forecast = shorten_array(Therm_forecast, step_size)
PV_forecast = shorten_array(PV_forecast, step_size)
# print(Elec_Demand_Forecast)
# print(PV_forecast)
# print(Elec_Demand_Forecast-PV_forecast)
#Electricity prices
# The subsequent calculations are to match the paper, first 6 hours and last 1 hour is lower cost.
cheap_am = int((24*60/4)/step_size)
regular = int((24*60*2/3)/step_size)
cheap_pm = int((24*60/12)/step_size)

Elec_Buy = np.concatenate((0.0575*np.ones(cheap_am), 0.0825*np.ones(regular), 0.0575*np.ones(cheap_pm+1))) 
Elec_Sell = Elec_Buy

# Elec_Sell = 0.0379 

#TODO importing electricity price, discuss with supervisor actual practicality of it 