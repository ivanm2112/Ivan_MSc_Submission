import pandas as pd
import numpy as np
import os

step_size=int(os.getenv("step_size"))
#step_size = 30
#Loading forecast excel file
data_file = pd.ExcelFile("./CREST_Demand_Model_v2.3.3.xlsm").parse("Results - aggregated")

#Cleaning up the file due to Excel formatting
data_file.columns = data_file.iloc[0]
data_file = data_file.drop([0,1,2])

#Loading individual forecasts
Elec_Demand_Forecast = 0*np.array(data_file["Total electricity demand"])
Therm_forecast = 0*np.array(data_file["Thermal demand for hot water"])
PV_forecast = 0*np.array(data_file["Total PV output"])   #1440 size

#Shortening the arrays to step size defined at the start.
from numpy_utils import shorten_array

Elec_Demand_Forecast = shorten_array(Elec_Demand_Forecast, step_size)
Therm_forecast = shorten_array(Therm_forecast, step_size)
PV_forecast = shorten_array(PV_forecast, step_size)


#Therm_forecast[47]=2

#Electricity prices
# The subsequent calculations are to match the paper, first 6 hours and last 1 hour is lower cost.

cheap_am = int((24*60/4)/step_size)
regular = int((24*60*2/3)/step_size)
cheap_pm = int((24*60/12)/step_size)

Elec_Buy = np.concatenate((1*np.ones(12), 1000*np.ones(36), 1000*np.ones(1))) 
Elec_Sell = Elec_Buy

# Elec_Sell = 0.0379 
#print(Elec_Demand_Forecast)
#print(Therm_forecast)
#print(PV_forecast)