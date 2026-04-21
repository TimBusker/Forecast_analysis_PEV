# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:23:22 2022

@author: tbr910
"""


#%% import packages  
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset as netcdf_dataset
from pylab import *
import urllib.request
import datetime
import re
import random
import calendar
import geopandas
import regionmask
import xskillscore as xs
import warnings
import cfgrib
import sys

# Library
import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pandas as pd
# Define path
path = '/scistor/ivm/tbr910/Forecast_action_analysis'

os.chdir(path)
import function_def
from function_def import plot_xarray_dataset
from function_def import plot_xr_facet_seas5

#%% NOTES 
#- seas5 files are in 'SEAS5_RAW_V2' folder. But.. Download needs to be completed. Now 2000-2020 period is not used from this folder, but from 'OLD_RAW' INSTEAD 
#
#
#%% user input section 

#%%% enter r indicator & ROI
r_indicator = 'rainfall_totals'

ROI = 'HAD'


alternative_Fval= False ## Chose between two different Fval Equations. Yields same result


#%%% time vars
years_for_nc= (range(1981,2023))
months= (range(1,13))
months_z= []
for i in months: 
    j=f"{i:02d}"
    months_z.append(j)  

month_names=list(calendar.month_name) 

for i in years_for_nc: 
    print (i)

ECMWF_2000_onwards=False # Whether to do analysis from year 2000 onwards 

#%%%% define 'month_range_selector' start/end
start_month= 10
end_month=11



#%% import paths --> CHIRPS FROM TAMSAT OR CHIRPS FROM OFFICIAL WEBSITE
CHIRPS_00=path_folder = os.path.join(path, 'tamsat_alert_archive_data/historical/rainfall/CHIRPS') #--> deleted
CHIRPS_00_HAD=os.path.join(path, 'tamsat_alert_archive_data/historical/Rainfall_HAD/CHIRPS')# --> deleted 

#CHIRPSV2=os.path.join(path, 'CHIRPS025/Africa')
CHIRPSV2_HAD=os.path.join(path, 'CHIRPS025/HAD') 
#os.chdir(CHIRPSV2)



#%% month-selectors
def month_selector(month_select):
    return (month_select == month_of_interest)  ## can also be a range

def month_range_selector(month):
    return (month >= start_month) & (month <= end_month)
        

#%% Load and process rainfall data 

#%%% Download CHIRPS data 
# os.chdir(path)
# from function_def import download_data
# os.chdir(CHIRPSV2)
# download_data(1980, 2021, 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p25', CHIRPSV2)

#%%% Crop rainfall for HAD region 
# os.chdir(CHIRPSV2) ## unit mm/ time step 
# os.listdir()

# for i in os.listdir():## when using chirps, tamsat: [-4:]
#     os.chdir(CHIRPSV2)
#     rainfall=xr.open_dataset(i) 
#     rainfall=rainfall.rename({'x': 'longitude','y': 'latitude'}).drop('spatial_ref').squeeze('band').drop('band') # not necessary for TAMSAT
#     rainfall=rainfall.assign_coords(time=pd.Timestamp(i[12:-4])).expand_dims('time')# not necessary for TAMSAT
#     rainfall_HAD=rainfall.where((rainfall.latitude > -4.7) & (rainfall.latitude<14.9) & (rainfall.longitude > 33.0) &(rainfall.longitude < 51.4) , drop=True)
#     os.chdir(CHIRPSV2_HAD)
#     rainfall_HAD.to_netcdf("%s_sub_HAD.nc" %(i[:-4]))  #for tamsat prate_tamsat -11:-7  for chirps --> "chirps-v2_%s_sub_HAD.nc" %(i)[14:-3]



#%%% Open all files as one XR 
# os.chdir(CHIRPSV2_HAD)
# P_HAD= xr.open_mfdataset([i for i in os.listdir() if 'ALL' not in i],combine='nested', concat_dim='time') #chunks={'time':10} chunks={'time':1, 'longitude':73, 'latitude':79}
# P_HAD.to_netcdf("chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc")  #[-11:-7] for tamsat  prate_tamsat_sub_HAD_ALL_YEARS.nc

#%%% pre-process rainfall 
os.chdir(CHIRPSV2_HAD)
P_HAD=xr.open_dataset('chirps-v2_ALL_YEARS_sub_HAD_ORIGINAL.nc')#chirps-v2_ALL_YEARS_sub_HAD_NEW
P_HAD=P_HAD.rename(band_data='tp') #precip 
P_HAD_DN=P_HAD.where(P_HAD['tp']!=-9.999e+03) ## save -9999 values as NAN
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']<1e+10) ## delete all high values (becomes nan). ASK VICKY WHERE THESE HIGH VALUES COME FROM! 
P_HAD_DN=P_HAD_DN.where(P_HAD_DN['tp']>-2000) #Delete very small rainfall values

#P_HAD_DN=P_HAD_DN.where(P_HAD_DN.time.dt.year>1982).dropna(dim='time',how='all') # for TAMSAT, use >0 as some very low values are present 


#%%% Create land mask

#%%%% Open SEAS5 file  
os.chdir(os.path.join(path, 'ECMWF/SEAS5/OLD_RAW'))
seas5=xr.open_dataset('FINAL_ecmwf_seas5_2000-01-01.grib')

#%%%% Upscale CHIRPS to SEAS5 - create land and rainfall mask 
P_UP=P_HAD_DN.interp_like(seas5.tp, method='linear') #unit is mm/day #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

#%%%%% land mask 
total_rainfall_whole_period=P_UP.assign(total_rainfall=(P_UP.tp.sum(dim=['time']))) ## only to mask areas that never get rainfall 
land_mask=total_rainfall_whole_period.total_rainfall.where(total_rainfall_whole_period.total_rainfall ==0, 1)

#%%%%% OND rain mask 
from function_def import P_mask
OND_mask=P_mask(P_UP,9, 'QS-OCT', 10)

        
#%%%%% MAM rain mask. NOTE: Only OND and MAM are masked, not monthly totals 
MAM_mask=P_mask(P_UP,2, 'QS-MAR', 10)

#%%%% Optional: wet days convertion 
wet_days=P_UP.where(P_UP['tp'] >1, 0)
wet_days=wet_days.where(wet_days['tp'] ==0, 1)       
if r_indicator=='wet_days':  
    P_UP=wet_days 
        
        

#%%% Resample to months 
time_interval= 'MS'
P_UP=P_UP.resample(time=time_interval).sum() ## monthly 


#%%%%% land mask
P_UP=P_UP.where(land_mask==1,np.nan)


#%% Calculate climatology (mean, high and low tercile)
# clim_mean= P_UP.groupby("time.month").mean("time")
# clim_lo_thres= P_UP.groupby("time.month").quantile(0.33, dim='time')
# clim_up_thres=P_UP.groupby("time.month").quantile(0.67, dim='time')




#%%% Visualize clim per month 
# for i in range(0,12):
#     month=clim_mean.isel(month=i)
#     plot_xarray_dataset(month,'tp', 'latitude', 'longitude','%s'%(calendar.month_name[i+1]), 1, 'none')
#     month_wet=clim_up_thres.isel(month=i)
#     plot_xarray_dataset(month_wet,'tp', 'latitude', 'longitude','  for month %s'%(i+1), 1, 'none')
#     month_dry=clim_lo_thres.isel(month=i)
#     plot_xarray_dataset(month_dry,'tp', 'latitude', 'longitude','TAMSAT climatology dry for month %s'%(i+1), 1, 'none')

# clim_mean_seas= P_UP.groupby("time.season").mean("time")
# for i in range(0,4):
#     season=clim_mean_seas.isel(season=i)    
#     plot_xarray_dataset(season,'tp', 'latitude', 'longitude','P clim mean for season %s'%(i+1), 1, 'none')
    

#%% Forecast validation 

#%%% Deaccumulate forecasts (speeds up script)


# os.chdir(os.path.join(path, 'ECMWF/SEAS5/SEAS5_RAW_V2'))

# for i in os.listdir(): 
#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/SEAS5_RAW_V2'))
    
#     if i[-3:]!='idx':   
#         if '99' in i: ## DE-ACCUMULATE ONLY 1990 -2000 
#             print (i[12:-5])
#             seas5=xr.open_dataset(i)
#             seas5_diff=seas5.diff(dim='step') ## deaccumulate
#             seas5_diff=seas5_diff.swap_dims({"step": "valid_time"})
#             os.chdir(os.path.join(path, 'ECMWF/SEAS5/DA'))
#             seas5_diff.to_netcdf('FINAL_ecmwf_seas5_'+i[12:-5]+'_DA.nc')





#%%% interpolate 1981-2000 files to match 2000-2020 files 
# NOTE: interpolation goes wrong when trying to interpolate with same dim names (e.g. interpolate one ECMWF dataset to another one)
# SAVES AS FLOAT 64, NEXT TIME SAVE AS FLOAT32 TO REDUCE DATA SIZE
#only diff: new ds in float64 instead of float32
# os.chdir(os.path.join(path, 'ECMWF/SEAS5/DA'))

# for i in os.listdir()[50:228]: # first 20 years  

#     x=xr.open_dataset(i)
#     x_downscaled=x.interp_like(P_UP.tp, method= 'linear')
#     x.close()
#     x_downscaled.to_netcdf(i)



#%%% Monthly obs-for observation file
## Here, tp_obs and tp_for will be created as variables, with the dims 'valid_time' and 'time'. 
## tp_obs: time dimension is not explicitly used here. Time is indicatated with var valid_time 
## tp_for: time is the initiation time of the forecast, valid_time the forecasted month/season 

# os.chdir(os.path.join(path, 'ECMWF/SEAS5/DA')
##warnings.filterwarnings('ignore')


#%%%% obs-for loop 
# for y in years_for_nc: 
#     merged_obs_for_master=xr.Dataset()
#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/DA'))
#     for m in months_z: # select forecast initiation  
#         merged_obs_for=xr.Dataset()
#         print ('Forecast date is: %s %s'%(y,m))
#         seas5=xr.open_dataset('FINAL_ecmwf_seas5_%s-%s-01_DA.nc'%(y, m)) 
#         seas5=seas5*1000 ## rainfall (tp) unit is m! https://apps.ecmwf.int/codes/grib/param-db/?id=228
        
#         # wet days 
#         seas5_wet_days=seas5.where(seas5['tp'] >1, 0)
#         seas5_wet_days=seas5_wet_days.where(seas5_wet_days['tp'] ==0, 1)       
#         if r_indicator=='wet_days':  
#             seas5=seas5_wet_days 
                
        
#         seas5=seas5.resample(valid_time=time_interval).sum() ## resample seas5 to monthly
#         seas5_mean=seas5.drop('surface')#.mean(dim='number') ## select or de-selct this one to switch between deterministic or probabilistic  ## .isel(valid_time=slice(0, 7)) --> possible to select first time steps in lead
#         seas5_mean=seas5_mean.rename(tp='tp_for') ## create variable for tp forecasted
        
#         for i in range(0,len(seas5_mean.valid_time)): ## loop over all leads
#             valid_date=seas5_mean.valid_time[i].values
#             print(valid_date)
#             initiation_date= seas5_mean.time.values
#             lead_time= int(round((float(valid_date-initiation_date)/(2592000000000000)),0))
            
#             seas5_vt=seas5_mean.sel(valid_time=str(valid_date)[0:10]).expand_dims('valid_time')## select seas5 rainfall for specific month        
#             tp_obs=P_UP.sel(time=valid_date) ## select tamsat/chirps rainfall for specific month 
#             tp_obs=tp_obs.assign_coords({"valid_time":valid_date})
#             tp_obs=tp_obs.expand_dims('valid_time')
#             tp_obs=tp_obs.drop('time')
#             tp_obs=tp_obs.rename(tp='tp_obs')## create variable for tp observed
            
#             obs_for= xr.merge([seas5_vt,tp_obs])
#             obs_for= obs_for.expand_dims('time')
#             merged_obs_for= xr.merge([merged_obs_for,obs_for])

#         merged_obs_for_master= xr.merge([merged_obs_for_master,merged_obs_for])

#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981') 

#     if r_indicator=='wet_days': 
#         os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981_wd'))
#     merged_obs_for_master.to_netcdf('obs_for_dataset_%s_%s_ENS.nc'%(r_indicator,y))    


#%%%% merge year obs-for files to one big netcdf 
# os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981'))

# if r_indicator=='wet_days': 
#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981_wd'))

# merged_obs_for= []

# for i in os.listdir(): 
#     if 'ENS.nc' in i: 
#         merged_obs_for.append(i)
#         convertion= xr.open_dataset(i)
#         convertion1=convertion.astype(float32)
#         convertion.close()
#         convertion1.to_netcdf(i)
        
#         print(i)

#%%%% save obs-for netcdf   
# seas5_merged= xr.open_mfdataset(merged_obs_for,combine='nested', concat_dim='time',chunks={'time':6, 'longitude':45, 'latitude':49, 'valid_time':8, 'number':5})
# seas5_merged.to_netcdf('obs_for_dataset_ENS_monthly2.nc')


#%% Calculate scores 

#%%%% Load obs-for file 
os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981'))
obs_for=xr.open_dataset('obs_for_dataset_ENS_monthly2.nc').load()

if r_indicator=='wet_days': 
    os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981_wd'))
    obs_for=xr.open_dataset('obs_for_dataset_ENS_monthly_wd.nc')

if ECMWF_2000_onwards==True:
    obs_for=obs_for.where(obs_for.tp_obs.valid_time.dt.year>1999).dropna(dim='valid_time', how='all')

obs_for=obs_for.assign(total_rainfall=(obs_for.tp_obs.sum(dim=['valid_time', 'time']))) ## only to mask areas that never get rainfall

#land_mask=land_mask.interp_like(obs_for.tp_obs, method= 'linear') #weird, but this is necessary to make the land_mask lat dim same as obs_for lat dim. They are the same but still xarray says they are not (maybe a digit problem)? 

obs_for=obs_for.assign_coords({"latitude": land_mask.latitude.values}) ## somehow necessary because the coordinates of obs_for are being read differently with the xarray version on the cluster 
obs_for=obs_for.assign_coords({"longitude": land_mask.longitude.values})

#obs_for=obs_for.interp_like(land_mask, method= 'linear')
obs_for['tp_obs']=obs_for['tp_obs'].where(land_mask ==1, np.nan) ## mask obs precip == 0 


#%%%% Set time interval for validation 
validation_length= 'seasonal' 

if validation_length=='seasonal':
    TOI_list= ['MAM', 'OND']#'MAM', 'OND'

else: 
    TOI_list= months_z
    


#%%%% Set region & mask 
name_list=['Kenya','Ethiopia','Somalia']
county_sf = geopandas.read_file((os.path.join(path, 'vector_data/african_countries/afr_g2014_2013_0.shp')))

longitude = obs_for.longitude.values
latitude = obs_for.latitude.values
mask = regionmask.mask_geopandas(county_sf,longitude,latitude)
mask=mask.rename({'lon': 'longitude','lat': 'latitude'})

# all datasets which are used later need to be included (Kenia, Somalia and Ethiopia)
if ROI=='Kenya': 
    ##### mask xarray 
    ID=county_sf.index[county_sf['ADM0_NAME']==name_list[0]].tolist()
    ID = int(ID[0])
    
    obs_for= obs_for.where(mask==ID, np.nan)
    OND_mask= OND_mask.where(mask==ID, np.nan)
    MAM_mask= MAM_mask.where(mask==ID, np.nan)
    land_mask= land_mask.where(mask==ID, np.nan)
    land_mask=land_mask.where(land_mask==1, np.nan)
        
    
if ROI=='Ethiopia': 
    ##### mask xarray 
    ID=county_sf.index[county_sf['ADM0_NAME']==name_list[1]].tolist()
    ID = int(ID[0])
    
    obs_for= obs_for.where(mask==ID, np.nan)
    OND_mask= OND_mask.where(mask==ID, np.nan)
    MAM_mask= MAM_mask.where(mask==ID, np.nan)
    land_mask= land_mask.where(mask==ID, np.nan)
    land_mask=land_mask.where(land_mask==1, np.nan)
        
    
if ROI=='Somalia': 
    ##### mask xarray 
    ID=county_sf.index[county_sf['ADM0_NAME']==name_list[2]].tolist()
    ID = int(ID[0])
    
    obs_for= obs_for.where(mask==ID, np.nan) 
    OND_mask= OND_mask.where(mask==ID, np.nan)
    MAM_mask= MAM_mask.where(mask==ID, np.nan)
    land_mask= land_mask.where(mask==ID, np.nan)
    land_mask=land_mask.where(land_mask==1, np.nan)
        
    

if ROI== 'HAD': 
    ##### mask xarray 
    ID_list=[]
    
    for name in name_list:     
        ID=county_sf.index[county_sf['ADM0_NAME']==name].tolist()
        ID = int(ID[0])
        ID_list.append(ID)    
        
    obs_for= obs_for.where(((mask==ID_list[0]) | (mask==ID_list[1]) | (mask==ID_list[2])), np.nan)
    OND_mask= OND_mask.where(((mask==ID_list[0]) | (mask==ID_list[1]) | (mask==ID_list[2])), np.nan)
    MAM_mask= MAM_mask.where(((mask==ID_list[0]) | (mask==ID_list[1]) | (mask==ID_list[2])), np.nan)
    land_mask= land_mask.where(((mask==ID_list[0]) | (mask==ID_list[1]) | (mask==ID_list[2])), np.nan)
    land_mask=land_mask.where(land_mask==1, np.nan)
    
#%%% Pearson's correlation 
# obs_for_determ= obs_for.mean(dim='number') ## mean of the ensembles   
# for TOI in TOI_list:
            
#     def month_selector(month_select):
#         return (month_select == month_of_interest)  ## can also be a range
    
# #%%%% select months/seasons
#     if TOI=='OND':## select season of interest (indicated by start month)
#         obs_for_season=obs_for_determ.resample(valid_time='QS-OCT').sum()
#         month_of_interest= int(months_z[9]) ## oct ## convert to int to select month, doesnt work with string
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
#         valid_time_sel=valid_time_sel.where(OND_mask.tp==1, np.nan) # mask using OND mask 

#     elif TOI=='MAM': ## select season of interest (indicated by start month)
#         obs_for_season=obs_for_determ.resample(valid_time='QS-MAR').sum()
#         month_of_interest= int(months_z[2]) ## march
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
#         valid_time_sel=valid_time_sel.where(MAM_mask.tp==1, np.nan) # Mask using MAM mask 
#     elif TOI== 'ON':  
#         valid_time_sel = obs_for_determ.sel(valid_time=month_range_selector(obs_for_determ['valid_time.month']))
#         valid_time_sel=valid_time_sel.resample(valid_time='2MS').sum()#.dropna(dim='valid_time', how='all')
#         month_of_interest= int(months_z[9])
#         valid_time_sel=valid_time_sel.sel(valid_time=month_selector(valid_time_sel['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate     
#     else: ## select month of interest 
#         month_of_interest=int(TOI) 
#         valid_time_sel = obs_for_determ.sel(valid_time=month_selector(obs_for_determ['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate

# #%%%% select lead

#     correlation_dataset=xr.Dataset()
#     g=0
#     ## select specific month 

#     leads=[0,1,2,3,4,5,6]# ,1,2,3,4,5,6
#     p_corr_merged=xr.Dataset()
    
   
#     for l in leads:
#         correlation_dataset=xr.Dataset()
#         g=0
#         for i in range(0,len(valid_time_sel.valid_time)):
#             sel_vt= valid_time_sel.isel(valid_time=i)
            
#             for j in range(0,len(sel_vt.time)):
#                 sel_it=sel_vt.isel(time=j)
#                 sel_it=sel_it.expand_dims('valid_time')
                
#                 lead=int(round((float(sel_it.lead.values)/(2592000000000000)),0))#int(sel_it.lead_month_stamp.values)
#                 if lead==l:
#                     if g==0:
#                         correlation_dataset=xr.merge([correlation_dataset,sel_it])
#                     else:
#                         correlation_dataset=xr.combine_by_coords([correlation_dataset,sel_it])
#                 g=g+1

#         correlation_dataset=correlation_dataset.assign_coords(lead=((np.median(correlation_dataset.lead.values))))

# #%%%% calculate corr per lead 
#         p_corr= xs.pearson_r(correlation_dataset.tp_obs, correlation_dataset.tp_for, dim='valid_time').to_dataset(name='corr')
#         p_corr['p_value']= xs.pearson_r_p_value(correlation_dataset.tp_obs, correlation_dataset.tp_for, dim='valid_time').assign_coords(lead=((np.median(correlation_dataset.lead.values))))
#         p_corr['p_value_binary']=p_corr['p_value']<0.05
#         p_corr=p_corr.expand_dims('lead')

# #%%%% create merged corr file   
    
#         if l==0:
#             p_corr_merged=xr.merge([p_corr_merged,p_corr])
        
#         else: 
#             p_corr_merged=xr.combine_by_coords([p_corr_merged,p_corr]) ## skill specific month (all leads). 
    
   
#     if validation_length=='seasonal':
#         plot_xr_facet_seas5(p_corr_merged,'corr', -1,1, TOI, mpl.cm.RdYlBu_r, 'Pearson correlation coefficient (r)', 'Pearson correlation coefficient (r) for %s'%(TOI))
#     else: 
#         plot_xr_facet_seas5(p_corr_merged,'corr', -1,1, TOI,mpl.cm.RdYlBu_r, 'Pearson correlation coefficient (r)', 'Pearson correlation coefficient (r) for %s'%(month_names[int(TOI)]))

#     plt.close()


# #%%% Mean absolute error (MAE) 
# obs_for_determ= obs_for.mean(dim='number') ## mean of the ensembles  
# for TOI in TOI_list:

# #%%%% select months/seasons            
#     def month_selector(month_select):
#         return (month_select == month_of_interest)  ## can also be a range
        
#     ############################################# select months/season ############################
#     if TOI=='OND':## select season of interest (indicated by start month)
#         obs_for_season=obs_for_determ.resample(valid_time='QS-OCT').sum()
#         month_of_interest= int(months_z[9]) ## oct ## convert to int to select month, doesnt work with string
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
        
#         valid_time_sel=valid_time_sel.where(OND_mask.tp==1, np.nan) # mask using OND mask 
        
#     elif TOI=='MAM': ## select season of interest (indicated by start month)
#         obs_for_season=obs_for_determ.resample(valid_time='QS-MAR').sum()
#         month_of_interest= int(months_z[2]) ## march
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
        
#         valid_time_sel=valid_time_sel.where(MAM_mask.tp==1, np.nan) # Mask using MAM mask 
        
#     elif TOI== 'ON':  
#         valid_time_sel = obs_for_determ.sel(valid_time=month_range_selector(obs_for_determ['valid_time.month']))
#         valid_time_sel=valid_time_sel.resample(valid_time='2MS').sum()#.dropna(dim='valid_time', how='all')
#         month_of_interest= int(months_z[9])
#         valid_time_sel=valid_time_sel.sel(valid_time=month_selector(valid_time_sel['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate    

        
#     else: ## select month of interest 
#         month_of_interest=int(TOI) 
#         valid_time_sel = obs_for_determ.sel(valid_time=month_selector(obs_for_determ['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate

# #%%%% select lead
#     mae_dataset=xr.Dataset()
#     g=0
#     leads=[0,1,2,3,4,5,6]
#     mae_merged=xr.Dataset()    
        
#     for l in leads:
#         mae_dataset=xr.Dataset()
#         g=0
#         for i in range(0,len(valid_time_sel.valid_time)):
#             sel_vt= valid_time_sel.isel(valid_time=i)
            
#             for j in range(0,len(sel_vt.time)):
#                 sel_it=sel_vt.isel(time=j)
#                 sel_it=sel_it.expand_dims('valid_time')
                
#                 lead=int(round((float(sel_it.lead.values)/(2592000000000000)),0))#int(sel_it.lead_month_stamp.values)
#                 if lead==l:
#                     if g==0:
#                         mae_dataset=xr.merge([mae_dataset,sel_it])
#                     else:
#                         mae_dataset=xr.combine_by_coords([mae_dataset,sel_it])
#                 g=g+1
                
# #%%%% calculate mae per lead
#         mae_dataset=mae_dataset.assign_coords(lead=((np.median(mae_dataset.lead.values))))
#         mae= xs.me(mae_dataset.tp_for, mae_dataset.tp_obs, dim='valid_time').to_dataset(name='mae').expand_dims('lead') #a,b a-b 
#         if l==0:
#             mae_merged=xr.merge([mae_merged,mae])
        
#         else: 
#             mae_merged=xr.combine_by_coords([mae_merged,mae]) ## skill specific month (all leads). 
    
    
# #%%%% create merged mae file 
#     mae_merged=mae_merged.where(mae_merged.mae!=0, np.nan)
#     if validation_length=='seasonal':   
#         plot_xr_facet_seas5(mae_merged,'mae', -150,150, TOI, mpl.cm.RdYlBu_r, 'Mean  Error (ME) in mm/season', 'Mean Absolute Error (MAE) for season %s'%(TOI))
#     else: 
#         plot_xr_facet_seas5(mae_merged,'mae', -150,150, TOI, mpl.cm.RdYlBu_r, 'Mean  Error (ME) in mm/season', 'Mean Absolute Error (MAE) for %s'%(month_names[int(TOI)]))
#     plt.close()


# #%%% CRPSS  
# os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981'))
# obs_for=xr.open_dataset('obs_for_dataset_ENS_monthly2.nc')
# obs_for['tp_obs']=obs_for['tp_obs'].where(land_mask ==1, np.nan) 
# ensemble_number= 25 

 
# for TOI in TOI_list:
# #%%%% select months/seasons           
#     def month_selector(month_select):
#         return (month_select == month_of_interest)  ## can also be a range
    
#     if TOI=='OND':## select season of interest (indicated by start month)
#         obs_for_season=obs_for.resample(valid_time='QS-OCT').sum()
#         month_of_interest= int(months_z[9]) ## oct ## convert to int to select month, doesnt work with string
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
        
#         valid_time_sel=valid_time_sel.where(OND_mask.tp==1, np.nan) # mask using OND mask 
        
#     elif TOI=='MAM': ## select season of interest (indicated by start month)
#         obs_for_season=obs_for.resample(valid_time='QS-MAR').sum()
#         month_of_interest= int(months_z[2]) ## march
#         valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate  
        
#         valid_time_sel=valid_time_sel.where(MAM_mask.tp==1, np.nan) # mask using MAM mask 
#     elif TOI== 'ON':  
#         valid_time_sel = obs_for.sel(valid_time=month_range_selector(obs_for['valid_time.month']))
#         valid_time_sel=valid_time_sel.resample(valid_time='2MS').sum()#.dropna(dim='valid_time', how='all')
#         month_of_interest= int(months_z[9])
#         valid_time_sel=valid_time_sel.sel(valid_time=month_selector(valid_time_sel['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate    

        
#     else: ## select month of interest 
#         month_of_interest=int(TOI) 
#         valid_time_sel = obs_for.sel(valid_time=month_selector(obs_for['valid_time.month']))
#         valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate

#     counter2=0
     
#     ## compute quantiles
#     quantiles= []
#     for i in range(1,ensemble_number+1): 
#         print (i)
#         quantile= i/(ensemble_number+1)
#         print (quantile)
#         quantiles.append(quantile)
                           
    
# #%%%% select lead
    
#     g=0
#     leads=[0,1,2,3,4,5,6]
#     CRPSS_merged=xr.Dataset()    
        
#     for l in leads:
#         obs_for_lead_dataset=xr.Dataset()
#         g=0
#         for i in range(0,len(valid_time_sel.valid_time)):
#             sel_vt= valid_time_sel.isel(valid_time=i)
            
#             for j in range(0,len(sel_vt.time)):
#                 sel_it=sel_vt.isel(time=j)
#                 sel_it=sel_it.expand_dims('valid_time')
                
#                 lead=int(round((float(sel_it.lead.values)/(2592000000000000)),0))#int(sel_it.lead_month_stamp.values)
#                 if lead==l:
#                     if g==0:
#                         obs_for_lead_dataset=xr.merge([obs_for_lead_dataset,sel_it])
#                     else:
#                         obs_for_lead_dataset=xr.combine_by_coords([obs_for_lead_dataset,sel_it])
#                 g=g+1
#         #%%%% monthly means >10mm 
#         obs_for_lead_dataset=obs_for_lead_dataset.assign_coords(lead=((np.median(obs_for_lead_dataset.lead.values))))
#         obs_for_lead_dataset=obs_for_lead_dataset.assign(monthly_means=(obs_for_lead_dataset.tp_obs.mean(dim=['valid_time'])))
#         obs_for_lead_dataset=obs_for_lead_dataset.where(obs_for_lead_dataset['monthly_means'] > 10)
#         obs_for_lead_dataset=obs_for_lead_dataset.assign(quantiles=(obs_for_lead_dataset["tp_obs"].quantile(quantiles, dim='valid_time')))
#         obs_for_lead_dataset=obs_for_lead_dataset.expand_dims('surface')
#         #%%%% calculate CRPSS 
#         CRPS_benchmark=xs.crps_ensemble(obs_for_lead_dataset.tp_obs, obs_for_lead_dataset.quantiles, member_dim='quantile', dim='valid_time')#.rename(tp='CRPS_benchmark')              
#         CRPS=xs.crps_ensemble(obs_for_lead_dataset.tp_obs, obs_for_lead_dataset.tp_for, member_dim='number', dim='valid_time')
#         CRPSS= 1-(CRPS/CRPS_benchmark) 
#         CRPSS=CRPSS.to_dataset(name='CRPSS').expand_dims('lead')

#         #%%%% Create merged CRPSS file         
#         if l==0:
#             CRPSS_merged=xr.merge([CRPSS_merged,CRPSS])
        
#         else: 
#             CRPSS_merged=xr.combine_by_coords([CRPSS_merged,CRPSS]) ## skill specific month (all leads). 

    
#     CRPSS_merged=CRPSS_merged.squeeze('surface') #(only needed for CRPSS)

#     if validation_length=='seasonal':
#         plot_xr_facet_seas5(CRPSS_merged,'CRPSS', 0,0.4, TOI, plt.cm.get_cmap('coolwarm', 8), 'CRPSS', 'Cumulative Ranked Probability Skill Score (CRPSS) for %s'%(TOI))
    
#     else: 
#         plot_xr_facet_seas5(CRPSS_merged,'CRPSS', 0,0.4, TOI, plt.cm.get_cmap('coolwarm', 8), 'CRPSS', 'Cumulative Ranked Probability Skill Score (CRPSS) for %s'%(month_names[int(TOI)]))
#     plt.close()


#%%% ROC scores 
drought_quantiles=[0.33] # 0.1th quantile is excluded because sample size is to small (4 drought events) 
ensemble_number=25

#TOI_list=['OND']
for selected_quantile in drought_quantiles: 
    for TOI in TOI_list:
        
#%%%% select months/seasons         
        def month_selector(month_select):
            return (month_select == month_of_interest)  ## can also be a range

        ############################################# select months/season ############################
        if TOI=='OND':## select season of interest (indicated by start month)
            obs_for_season=obs_for.resample(valid_time='QS-OCT').sum()
            month_of_interest= int(months_z[9]) ## oct ## convert to int to select month, doesnt work with string
            valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
            valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate
            valid_time_sel=valid_time_sel.where(OND_mask.tp==1, np.nan) # mask using OND mask 
            
        elif TOI=='MAM': ## select season of interest (indicated by start month)
            obs_for_season=obs_for.resample(valid_time='QS-MAR').sum()
            month_of_interest= int(months_z[2]) ## march
            valid_time_sel = obs_for_season.sel(valid_time=month_selector(obs_for_season['valid_time.month']))
            valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate      
            valid_time_sel=valid_time_sel.where(MAM_mask.tp==1, np.nan) # mask using MAM mask 
            
        elif TOI== 'ON':
 
            valid_time_sel = obs_for.sel(valid_time=month_range_selector(obs_for['valid_time.month']))
            valid_time_sel=valid_time_sel.resample(valid_time='2MS').sum()#.dropna(dim='valid_time', how='all')
            month_of_interest= int(months_z[9])
            valid_time_sel=valid_time_sel.sel(valid_time=month_selector(valid_time_sel['valid_time.month']))
            valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate    


        else: ## select month of interest 
            month_of_interest=int(TOI) 
            valid_time_sel = obs_for.sel(valid_time=month_selector(obs_for['valid_time.month']))
            valid_time_sel=valid_time_sel.assign_coords(lead=((valid_time_sel.valid_time-valid_time_sel.time))) ## assign LT coordinate
        
#%%%% select lead            
        leads= [0,1,2,3,4,5,6]
        ROC_merged=xr.Dataset()    
        Fval_merged= xr.Dataset()
        g=0
        for l in leads:
            obs_for_lead_dataset=xr.Dataset()
            g=0
    
            for i in range(0,len(valid_time_sel.valid_time)):
                sel_vt= valid_time_sel.isel(valid_time=i)
                
                for j in range(0,len(sel_vt.time)):
                    sel_it=sel_vt.isel(time=j)
                    sel_it=sel_it.expand_dims('valid_time')
                    
                    lead=int(round((float(sel_it.lead.values)/(2592000000000000)),0))#int(sel_it.lead_month_stamp.values)
                    if lead==l:
                        if g==0:
                            obs_for_lead_dataset=xr.merge([obs_for_lead_dataset,sel_it])
                        else:
                            obs_for_lead_dataset=xr.combine_by_coords([obs_for_lead_dataset,sel_it])
                    g=g+1
            
            obs_for_lead_dataset=obs_for_lead_dataset.assign_coords(lead=((np.median(obs_for_lead_dataset.lead.values))))
            
            #obs_for_lead_dataset=obs_for_lead_dataset.assign(monthly_means=(obs_for_lead_dataset.tp_obs.mean(dim=['valid_time'])))
            #obs_for_lead_dataset=obs_for_lead_dataset.where(obs_for_lead_dataset['monthly_means'] > 10)
            

        
#%%%% ECMWF forecast probabilities 
            #%%%%% ECMWF ENS CLIM (S-M CLIMATE) --> https://confluence.ecmwf.int/display/FUG/S-M-climate%2C+the+Seasonal+Model+Climate
            obs_for_lead_dataset['clim_ens_thresholds']=obs_for_lead_dataset.tp_for.quantile(drought_quantiles, dim=('valid_time', 'number')) 
            #%%%%% Drought forecast flag per ENS
            obs_for_lead_dataset['drought_ens_boolean']=obs_for_lead_dataset.tp_for<obs_for_lead_dataset.clim_ens_thresholds       
            #%%%%% drought ENS members / all ens members 
            obs_for_lead_dataset['probability']=obs_for_lead_dataset.drought_ens_boolean.sum(dim='number')/ensemble_number
            
            ##### CHECK: NEGATIVE PRECIP VALUES IN ENSEMBLE?? obs_for_lead_dataset.tp_for.isel(valid_time=1).isel(latitude=1).isel(longitude=1)

#%%%% Drought occurance             
            obs_for_lead_dataset['clim_thresholds']= obs_for_lead_dataset.tp_obs.quantile(drought_quantiles, dim=('valid_time'))
            obs_for_lead_dataset['drought_boolean']=obs_for_lead_dataset.tp_obs<obs_for_lead_dataset.clim_thresholds
            
            obs_for_lead_dataset=obs_for_lead_dataset.where(land_mask ==1, np.nan) ## mask obs precip == 0. The binary classification above (to create drougth ens boolean) removed the nan values 
            
#%%%% Calculate ROC score               
            ROC=xs.roc(obs_for_lead_dataset.drought_boolean, obs_for_lead_dataset.probability, bin_edges=np.arange(0,1.1,0.1), dim='valid_time',return_results='all_as_metric_dim')#.sel(quantile=selected_quantile)#.rename(tp='CRPS_benchmark')              
            ROC=ROC.to_dataset(name='ROC').expand_dims('lead') ## otherwise the merge operation below doesn't function 
            if l==0:
                ROC_merged=xr.merge([ROC_merged,ROC])
            
            else: 
                ROC_merged=xr.combine_by_coords([ROC_merged,ROC]) ## skill specific month (all leads). 
           
#%%% Relative economic value


#%%%%% Clim freq            
            time_steps_total= obs_for_lead_dataset.drought_boolean.count(dim='valid_time').sel(quantile=selected_quantile) # how many time steps do we have in the data? 
            event_count=obs_for_lead_dataset.drought_boolean.sum(dim='valid_time').sel(quantile=selected_quantile)## how many times do we have a drought? 
            Climfreq=event_count/time_steps_total# Ratio of drought time steps compared to all time steps in rainfall dataset 
            (print('clim freq for %s season and %s th quantile rainfall = %s'%(TOI,selected_quantile, (event_count.where(land_mask==1).mean().values).round(0))))
            
#%%%% Contingency stats
##############################################                          Event occurs        Event doesn't occur       ##############################################
############################################## Early-warning                A  (HIT)                    B (FA)        ##############################################
##############################################                                                                        ##############################################
############################################## No Early - warnings          C  (MISS)                   D (CN)        ##############################################             
            
            p_counter=0
            Fval_merged_p= xr.Dataset()         
            ## probability thresholds 
            p_thresholds= [0.1,0.33,0.66] 
            
            for p in p_thresholds: 
                c_counter=0
                Fval_merged_CL= xr.Dataset() 
                
                cont_metrics= xs.Contingency(obs_for_lead_dataset.drought_boolean, obs_for_lead_dataset.probability,observation_category_edges =np.array([0.0, 0.99999, 1]), forecast_category_edges =np.array([0, p, 1]), dim='valid_time') ## this triggers with probability threshold of 33%. 
                
                
                ## Hits, misses, false alarms 
                hits=cont_metrics.hits().sel(quantile=selected_quantile)
                misses= cont_metrics.misses().sel(quantile=selected_quantile)
                false_alarms= cont_metrics.false_alarms().sel(quantile=selected_quantile)
                
                ## fraction of hits, misses, false alarms 
                hit_fraction= hits/time_steps_total
                miss_fraction= misses/time_steps_total
                false_alarm_fraction= false_alarms/time_steps_total
                ## hit rate
                hit_rate= cont_metrics.hit_rate().sel(quantile=selected_quantile) # hit rate= POD= hits/ (hits+misses) 
                ## False alarm rate 
                FAR= cont_metrics.false_alarm_rate().sel(quantile=selected_quantile)  # False alarm rate = FA / (FA+CN)      
                ## False alarm ratio 
                FAR_ratio= cont_metrics.false_alarm_ratio().sel(quantile=selected_quantile)# False alarm ratio = FA / (FA+HITS)
                
                
#%%%% set costs/ losses
                C_L_ratios= np.round(np.arange(0.01,1.00,0.01),2) # list of C/L ratios 
                L= 150 # protectable loss, in euro 
                
                C_protection= L*C_L_ratios ## list of protection costs to create C_L ratios above
                
#%%%% Calculate forecast value                
                for C,C_L in zip(C_protection, C_L_ratios):
                    
                    L_no_action= Climfreq*L # note: loss the same for each cell 
                    latitude=L_no_action.latitude.values
                    longitude=L_no_action.longitude.values
                    
                   
#%%%%% Step 1: Climatological expense (Ec)
                    Ec= np.minimum((L_no_action.values), C) ## climatological expense for the x percentile, calculated as numpy array
            
                    Ec = xr.DataArray( ## convert np array back to xarray 
                        data=Ec,
                    
                        dims=["latitude", "longitude"],
                    
                        coords=dict(
                    
                            longitude=(longitude),
                    
                            latitude=(latitude),
                    
                        ),
                    
                        attrs=dict(
                    
                            description="Ec",
                    
                        ),
            
                    )
                    
#%%%%% Step 2: perfect forecast expense (Epf)
                    Epf=Climfreq*C 
                    
#%%%%% Step 3: realistic forecast expense (Ef)
                    Ef= (hit_fraction*C)+ (false_alarm_fraction*C) + (miss_fraction*L)
                        

#%%%%% Step 4: forecast value calculation      
                    Fval= (Ec-Ef) / (Ec-Epf)
                    
                    
                    ## Alternative calcualtion of Fval 
                    if alternative_Fval==True: 
                                    
                        Ec = xr.DataArray( ## convert np array back to xarray 
                            data=Ec,
                        
                            dims=["latitude", "longitude"],
                        
                            coords=dict(
                        
                                longitude=(longitude),
                        
                                latitude=(latitude),
                        
                            ),
                        
                            attrs=dict(
                        
                                description="Ec",
                        
                            ),
                
                        )
                        
                        
                        Fval=  (np.minimum(C_L, Climfreq)-(FAR*(1-Climfreq)*C_L)+(hit_rate*Climfreq*(1-C_L))-Climfreq)/(np.minimum(C_L, Climfreq)-(Climfreq*C_L))
                    
                    Fval=Fval.to_dataset(name='Fval').assign_coords(C_L=C_L).expand_dims('lead').expand_dims('C_L') ## otherwise the merge operation below doesn't function 

                    #######################  CL Merger ################################
                    if c_counter==0:
                        Fval_merged_CL=xr.merge([Fval_merged_CL,Fval])
                    
                    else: 
                        Fval_merged_CL=xr.combine_by_coords([Fval_merged_CL,Fval]) ## skill specific month (all leads). 
        
                    c_counter=c_counter+1
                    
                #######################  P merger  ################################   
                Fval_merged_CL=Fval_merged_CL.assign_coords(p=p).expand_dims('p')
                
                if p_counter==0:
                    Fval_merged_p=xr.merge([Fval_merged_p,Fval_merged_CL])
                    
                else:
                    Fval_merged_p=xr.combine_by_coords([Fval_merged_p,Fval_merged_CL]) ## skill specific month (all leads).                 
                p_counter=p_counter+1
            
            #######################  L merger  ################################  
            if l==0:
                Fval_merged=xr.merge([Fval_merged,Fval_merged_p])
                    
            else:
                Fval_merged=xr.combine_by_coords([Fval_merged,Fval_merged_p]) ## skill specific month (all leads). 
        
        # MISTAKE! Fval_merged=Fval_merged.where(Fval_merged>0, 0) ## convert negative values to 0 
        Fval_merged= Fval_merged.where(land_mask==1, np.nan) ## mask using a land mask 
    
    
#%% PEV plot section
        
#%%% spatial PEV plot 
        fig=plt.figure(figsize=(23,20)) # H,W
        
        
        if validation_length== 'seasonal': 
            plt.suptitle('Economic value -- season: %s, severity threshold=%s'%(TOI, (str(selected_quantile*100)[:-2]+' %')), size=15, fontweight='bold')# (W,H)
        else: 
            plt.suptitle('Economic value -- month: %s, severity threshold=%s'%(month_names[int(TOI)], (str(selected_quantile*100)[:-2]+' %')), size=15, fontweight='bold')# (W,H)
    
#%%% PEV graphs 
        
#%%%% PEV Axes 
        gs=fig.add_gridspec(20,20,wspace=0.7,hspace=1.5)
        ax1=fig.add_subplot(gs[0:6,0:6]) # Y,X
        ax2=fig.add_subplot(gs[0:6,6:12],sharey=ax1)
        ax3=fig.add_subplot(gs[0:6,12:18],sharey=ax1)

        ax4=fig.add_subplot(gs[7:13,0:6])
        ax5=fig.add_subplot(gs[7:13,6:12])
        ax6=fig.add_subplot(gs[7:13,12:18])
        ax7=fig.add_subplot(gs[14:20,0:6])
        
#%%%% ROC Axes 
        ax11=fig.add_subplot(gs[0:3,0:3]) # Y,X
        ax22=fig.add_subplot(gs[0:3,6:9])
        ax33=fig.add_subplot(gs[0:3,12:15])

        ax44=fig.add_subplot(gs[7:10,0:3])
        ax55=fig.add_subplot(gs[7:10,6:9])
        ax66=fig.add_subplot(gs[7:10,12:15])
        ax77=fig.add_subplot(gs[14:17,0:3])        

    
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        plt.setp(ax6.get_yticklabels(), visible=False)        

#%%%% plot params 
        label_x= 'Cost-loss ratio'
        label_y= 'Potential economic value (PEV)' #V$_{ECMWFseas5}$
        
        label_x_size=25
        label_y_size= 25
        label_fontsize=25
        x_ticks=np.arange(0,0.9,0.2)#[0.0,0.4,0.8]
        y_ticks=np.arange(0.2, 0.7, 0.2)#[0.2, 0.6, 1.0] 0.2,1.1,0.2
        y_lim=1
        tick_size= 20
        title_size=30
        linewidth=5
        colors=['darkblue','red','darkgreen']   
    
    
 
#%%%% ax1
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0)
            Fval_lead=Fval_region.isel(lead=0)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
        
            ### plot 
            ax1.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
        

        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax1.set_title('lead=%s month'%(lead), size=title_size)   
        ## axis labels 
        ax1.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        ax1.set_ylabel(label_y, size=label_y_size, weight = 'bold')  
        #ax1.legend(fontsize=label_fontsize)    
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, y_lim])
        
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)
        ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        ax1.tick_params(axis='both', which='minor', labelsize=tick_size)
        
        

#%%%% ax2
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0)
            Fval_lead=Fval_region.isel(lead=1)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
            
            ### plot 
            ax2.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
            
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax2.set_title('lead=%s month'%(lead), size=title_size)   
        ## axis labels 
        ax2.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        #ax2.set_ylabel(label_y, size=20)  
        #ax2.legend(fontsize=label_fontsize)    
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, y_lim])  
        ax2.set_xticks(x_ticks)
        ax2.set_yticks(y_ticks)
        
        ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        ax2.tick_params(axis='both', which='minor', labelsize=tick_size)  
        
#%%%% ax3
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0)
            Fval_lead=Fval_region.isel(lead=2)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
        
            ### plot 
            ax3.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
            
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax3.set_title('lead=%s months'%(lead), size=title_size)   
        ax3.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        #ax3.set_ylabel(label_y, size=20)  
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, y_lim])     
        ax3.set_xticks(x_ticks)
        ax3.set_yticks(y_ticks)
        ax3.tick_params(axis='both', which='major', labelsize=tick_size)
        ax3.tick_params(axis='both', which='minor', labelsize=tick_size)
#%%%% ax4
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region=Fval_region.where(Fval_region.Fval>0, 0) ## convert negative values to 0 
            Fval_lead=Fval_region.isel(lead=3)
            C_L=Fval_lead.C_L.values
       
            Fval=Fval_lead.sel(p=p).Fval.values
       
            ### plot 
            ax4.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
           
           
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax4.set_title('lead=%s months'%(lead), size=title_size)   
        ## axis labels 
        ax4.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        ax4.set_ylabel(label_y, size=label_y_size, weight = 'bold')  
        #ax4.legend(fontsize=label_fontsize)    
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0,y_lim])  
        ax4.set_xticks(x_ticks)
        ax4.set_yticks(y_ticks)
        ax4.tick_params(axis='both', which='major', labelsize=tick_size)
        ax4.tick_params(axis='both', which='minor', labelsize=tick_size)
        
#%%%% ax5
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0)
            Fval_lead=Fval_region.isel(lead=4)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
        
            ### plot 
            ax5.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
            
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax5.set_title('lead=%s months'%(lead), size=title_size)   
        ## axis labels 
        ax5.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        #ax5.set_ylabel(label_y, size=20)  
        #ax5.legend(fontsize=label_fontsize)    
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0,y_lim])     
        ax5.set_xticks(x_ticks)
        ax5.set_yticks(y_ticks)
        ax5.tick_params(axis='both', which='major', labelsize=tick_size)
        ax5.tick_params(axis='both', which='minor', labelsize=tick_size)
        
#%%%% ax6
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0) 
            Fval_lead=Fval_region.isel(lead=5)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
        
            ### plot 
            ax6.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
            
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax6.set_title('lead=%s months'%(lead), size=title_size)   
        ## axis labels 
        ax6.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        #ax6.set_ylabel(label_y, size=20)  
        #ax6.legend(fontsize=label_fontsize)    
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0,y_lim])    
        ax6.set_xticks(x_ticks)
        ax6.set_yticks(y_ticks)
        ax6.tick_params(axis='both', which='major', labelsize=tick_size)
        ax6.tick_params(axis='both', which='minor', labelsize=tick_size)
#%%%% ax7
        for p in p_thresholds: 
            Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
            Fval_region.where(Fval_region.Fval>0,0)
            Fval_lead=Fval_region.isel(lead=6)
            C_L=Fval_lead.C_L.values
        
            Fval=Fval_lead.sel(p=p).Fval.values
        
            ### plot 
            ax7.plot(C_L, Fval, label='Probability threshold= %s'%(str(p*100)[:-2]) +'%',color=colors[p_thresholds.index(p)], linewidth=linewidth)
            
            
        lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        ax7.set_title('lead=%s months'%(lead), size=title_size)   
        ## axis labels 
        ax7.set_xlabel(label_x, size=label_x_size, weight = 'bold')
        ax7.set_ylabel(label_y, size=label_y_size, weight = 'bold')  
        #ax7.legend(fontsize=label_fontsize)    
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0,y_lim])      
        ax7.set_xticks(x_ticks)
        ax7.set_yticks(y_ticks)
        ax7.tick_params(axis='both', which='major', labelsize=tick_size)
        ax7.tick_params(axis='both', which='minor', labelsize=tick_size)
        
#%%% ROC sub-graphs

        #%%%% plot params 
        #1:1 benchmark line 
        x= np.arange(0,1.1,0.1)
        y=np.arange(0,1.1,0.1)
        
        label_x= (r'$FA_{rate}$')
        label_y= 'Hit rate' 
        label_x_size=25
        label_y_size= 25
        label_fontsize=25
        
        x_ticks=[0.2,0.8]
        y_ticks=[0.2,0.8]
        tick_size= 25
        title_size=60
        linewidth=5

        AUC_textsize= '17'
        
    
#%%%% ax11
        #%%%%% ROC stats 
        ROC_region=ROC_merged.mean(dim=('latitude','longitude'))
        ROC_lead=ROC_region.isel(lead=0).sel(quantile=selected_quantile) ## select quantile
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot 
        ax11.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax11.set_xticks(x_ticks)
        ax11.set_yticks(y_ticks)
        ax11.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax11.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax11.set_xlabel(label_x, size=label_x_size)
        ax11.set_ylabel(label_y, size=label_y_size)    
        ax11.yaxis.set_label_position("right")
        
        #%%%%% 1:1 benchmark line 
        ax11.plot(x,y, linestyle='--', color='red')
        
        #%%%%% AUC score 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax11.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
        
#%%%% ax22
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=1).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot 
        ax22.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax22.set_xticks(x_ticks)
        ax22.set_yticks(y_ticks)
        ax22.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax22.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right')     
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax22.set_xlabel(label_x, size=label_x_size)
        ax22.set_ylabel(label_y, size=label_y_size) 
        ax22.yaxis.set_label_position("right")
        #%%%%% 1:1 benchmark line  
        ax22.plot(x,y, linestyle='--', color='red')        
        
        #%%%%% AUC score 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax22.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))       

#%%%% ax33
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=2).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot
        ax33.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax33.set_xticks(x_ticks)
        ax33.set_yticks(y_ticks)
        ax33.plot(FPR, TPR)
        #%%%%% adjust labels
        for i, txt in enumerate(prob_labels):
            ax33.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)   
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax33.set_xlabel(label_x, size=label_x_size)
        ax33.set_ylabel(label_y, size=label_y_size)      
        ax33.yaxis.set_label_position("right")
        #%%%%% 1:1 benchmark line  
        ax33.plot(x,y, linestyle='--', color='red')
        #%%%%% AUC score 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax33.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))          

#%%%% ax44
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=3).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        
        #%%%%% plot
        ax44.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax44.set_xticks(x_ticks)
        ax44.set_yticks(y_ticks)
        ax44.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax44.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)     
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax44.set_xlabel(label_x, size=label_x_size)
        ax44.set_ylabel(label_y, size=label_y_size) 
        ax44.yaxis.set_label_position("right") 
        #%%%%% 1:1 benchmark line  
        ax44.plot(x,y, linestyle='--', color='red')
   

        #%%%%% AUC scores 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax44.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))         

#%%%% ax55
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=4).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot
        ax55.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax55.set_xticks(x_ticks)
        ax55.set_yticks(y_ticks)
        ax55.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax55.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)  
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0))) 
        ax55.set_xlabel(label_x, size=label_x_size)
        ax55.set_ylabel(label_y, size=label_y_size)   
        ax55.yaxis.set_label_position("right")
        #%%%%% 1:1 benchmark line  
        ax55.plot(x,y, linestyle='--', color='red')
        #%%%%% AUC scores 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax55.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

#%%%% ax66
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=5).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot
        ax66.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax66.set_xticks(x_ticks)
        ax66.set_yticks(y_ticks)
        ax66.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax66.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8) 
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax66.set_xlabel(label_x, size=label_x_size)
        ax66.set_ylabel(label_y, size=label_y_size)   
        ax66.yaxis.set_label_position("right") 
        #%%%%% 1:1 benchmark line 
        ax66.plot(x,y, linestyle='--', color='red')    

        #%%%%% AUC scores 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax66.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

#%%%% ax77
        #%%%%% ROC stats
        ROC_lead=ROC_region.isel(lead=6).sel(quantile=selected_quantile)
        TPR=ROC_lead.sel(metric='true positive rate').ROC.values
        FPR=ROC_lead.sel(metric='false positive rate').ROC.values
        prob_labels=np.round(ROC_lead.probability_bin.values, 2)
        #%%%%% plot 
        ax77.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
        ax77.set_xticks(x_ticks)
        ax77.set_yticks(y_ticks)
        ax77.plot(FPR, TPR)
        #%%%%% adjust labels 
        for i, txt in enumerate(prob_labels):
            ax77.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)         
        lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
        ax77.set_xlabel(label_x, size=label_x_size)
        ax77.set_ylabel(label_y, size=label_y_size) 
        ax77.yaxis.set_label_position("right")
        #%%%%% 1:1 benchmark line 
        ax77.plot(x,y, linestyle='--', color='red')    

        #%%%%% AUC score 
        AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
        ax77.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
        
#%%% Legend and show/save        
        ax7.legend(fontsize=label_fontsize, loc='lower right', bbox_to_anchor=(3, 0))    
        
        plt.show()
        fig.savefig(os.path.join(path, 'plots/python/PEV_%s.pdf')%(TOI), bbox_inches='tight')
        plt.close()
        
        

#%%% PEVmax plot 

        # label_x= 'C/L ratio'
        # label_y= 'V$_{ECMWFseas5}$'
        
        # fig=plt.figure()
        # Fval_region=Fval_merged.mean(dim=('latitude','longitude'))
        # F_max=Fval_region['Fval'].max(dim='p')
        
        # ## lead times 
        # for i in leads: 
        #     F_max_lead=F_max.isel(lead=i)
        #     plt.plot(C_L,F_max_lead,linewidth=2.0, linestyle='-', label= 'EVmax for lead %s' %(i))
        # plt.legend()
        
        
        # plt.title('Maximum economic value for %s'%(TOI))   
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.xlabel(label_x, size=12)
        # plt.ylabel(label_y, size=12) 
        # plt.tight_layout()
        # plt.show()
        # plt.close()        
        #%% SPATIAL ROC PLOT 
        ROC_merged=ROC_merged.sel(quantile=selected_quantile).sel(metric='area under curve').isel(probability_bin=1) ## select random probability bin, as the AUC is the same for every bin
        ROC_merged=ROC_merged.where(ROC_merged.ROC>=0.5,np.nan)
        if validation_length=='seasonal': 
            plot_xr_facet_seas5(ROC_merged,'ROC', 0.5,1, TOI, plt.cm.get_cmap('Reds',5), 'ROC-AUC', 'ROC curve for %s and severity threshold of %s'%(TOI, (str(selected_quantile*100)[:-2]+' %')))
        else: 
            plot_xr_facet_seas5(ROC_merged,'ROC', 0,1, TOI, plt.cm.get_cmap('coolwarm', 6), 'ROC-AUC', 'ROC curve for %s and severity threshold of %s'%((month_names[int(TOI)]), (str(selected_quantile*100)[:-2]+' %')))
        
        plt.close()   

        #%% SPATIAL PEVmax PLOT 
        Fval_max=Fval_merged.max(dim=('p', 'C_L'))
                
        if validation_length=='seasonal': 
            plot_xr_facet_seas5(Fval_max,'Fval', 0,1, TOI, plt.cm.get_cmap('Reds',5), 'PEVmax', 'PEVmax for %s and severity threshold of %s'%(TOI, (str(selected_quantile*100)[:-2]+' %')))
            
        else: 
            plot_xr_facet_seas5(Fval_max,'Fval', 0,1, TOI, plt.cm.get_cmap('Reds', 5), 'PEVmax', 'PEVmax for %s and severity threshold of %s'%((month_names[int(TOI)]), (str(selected_quantile*100)[:-2]+' %')))






        #%%% Summary stats 
        lt=0
        #%%%% WHOLE HAD 
        
        ROC_mean_HAD=ROC_merged.ROC.isel(lead=lt).mean(dim=('latitude', 'longitude')).values
        FVALmax_mean_HAD= Fval_max.isel(lead=lt).mean(dim=('latitude', 'longitude')).Fval.values
        print ('MEAN OF ROC AND PEVmax for HAD is %s and %s, respectively, for season %s'%(ROC_mean_HAD,FVALmax_mean_HAD, TOI))
        #%%%% KENYA
        ID=county_sf.index[county_sf['ADM0_NAME']==name_list[0]].tolist()
        ID = int(ID[0])
        ROC_kenya= ROC_merged.where(mask==ID, np.nan)
        ROC_mean_kenya=ROC_kenya.ROC.isel(lead=lt).mean(dim=('latitude', 'longitude')).values
        FVALmax_mean_kenya= Fval_max.isel(lead=lt).where(mask==ID, np.nan).mean(dim=('latitude', 'longitude')).Fval.values
        print ('MEAN OF ROC AND PEVmax for Kenya is %s and %s, respectively, for season %s'%(ROC_mean_kenya,FVALmax_mean_kenya, TOI))
        #%%%% SOMALIA 
        ID=county_sf.index[county_sf['ADM0_NAME']==name_list[2]].tolist()
        ID = int(ID[0])
        ROC_somalia=ROC_merged.where(mask==ID, np.nan)
        ROC_mean_somalia=ROC_somalia.ROC.isel(lead=lt).mean(dim=('latitude', 'longitude')).values
        FVALmax_mean_somalia= Fval_max.isel(lead=lt).where(mask==ID, np.nan).mean(dim=('latitude', 'longitude')).Fval.values
        print ('MEAN OF ROC AND PEVmax for Somalia is %s and %s, respectively, for season %s'%(ROC_mean_somalia,FVALmax_mean_somalia, TOI))        
        
        #%%%% ETHIOPIA 
        ID=county_sf.index[county_sf['ADM0_NAME']==name_list[1]].tolist()
        ID = int(ID[0])
        
        ## ROC 
        ROC_ethiopia=ROC_merged.where(mask==ID, np.nan)
        ROC_mean_ethiopia=ROC_ethiopia.ROC.isel(lead=lt).mean(dim=('latitude', 'longitude')).values
        
        ## Fval_max
        FVALmax_mean_ethiopia= Fval_max.isel(lead=lt).where(mask==ID, np.nan).mean(dim=('latitude', 'longitude')).Fval.values
        
        print ('MEAN OF ROC AND PEVmax for Ethiopia is %s and %s, respectively, for season %s'%(ROC_mean_ethiopia,FVALmax_mean_ethiopia, TOI))       
        
   
        
        #%%% LIVELIHOOD MASKS 
        
        
        # os.chdir(os.path.join(path, 'vector_data\\livelihood_zones_erin'))
        # longitude = Fval_max.longitude.values
        # latitude = Fval_max.latitude.values
        
        # #ap_mask
        # ap_mask = geopandas.read_file('livelihood_ap_2.shp')#.set_index("FNID") 
        # ap_mask = regionmask.mask_geopandas(ap_mask,longitude,latitude)
        # ap_mask=ap_mask.rename({'lon': 'longitude','lat': 'latitude'})     
        # ap_mask=ap_mask.to_dataset()
        # ap_mask=ap_mask.where(ap_mask.mask>=0, -9999)
        # ap_mask=ap_mask.where(ap_mask.mask==-9999, 1)        
        # ap_mask=ap_mask.where(ap_mask.mask==1,0)   


        # #p_mask
        # p_mask = geopandas.read_file('livelihood_p.shp')#.set_index("FNID") 
        # p_mask = regionmask.mask_geopandas(p_mask,longitude,latitude)
        # p_mask=p_mask.rename({'lon': 'longitude','lat': 'latitude'})     
        # p_mask=p_mask.to_dataset()
        # p_mask=p_mask.where(p_mask.mask>=0, -9999)
        # p_mask=p_mask.where(p_mask.mask==-9999, 1)        
        # p_mask=p_mask.where(p_mask.mask==1,0) 
        
        # #other_mask
        # other_mask = geopandas.read_file('livelihood_other.shp')#.set_index("FNID") 
        # other_mask = regionmask.mask_geopandas(other_mask,longitude,latitude)
        # other_mask=other_mask.rename({'lon': 'longitude','lat': 'latitude'})     
        # other_mask=other_mask.to_dataset()
        # other_mask=other_mask.where(other_mask.mask>=0, -9999)
        # other_mask=other_mask.where(other_mask.mask==-9999, 1)        
        # other_mask=other_mask.where(other_mask.mask==1,0)         
        
        # #ap-p mask 
        # app_mask= other_mask.where(other_mask.mask==0, np.nan) 
        # app_mask= app_mask.where(app_mask.mask!=0, 1)         
        # app_mask= app_mask.where(app_mask.mask==1, 0)         
        # app_mask= app_mask.where(land_mask==1, 0)         
        
        
        
        # Fval_merged_p
        
        
        

        # # c_mask = geopandas.read_file('livelihood_c.shp')#.set_index("FNID") 
        # # ##### mask xarray 
        # # ID=county_sf.index[county_sf['COUNTY']==county].tolist()
        # # ID = int(ID[0])
        # # ## NDVI
        # # NDVI_county= NDVI_P.where(mask==ID, np.nan)
        # # NDVI_county_mean=NDVI_county.mean(dim=('latitude', 'longitude'))        
        # # ## NDVI range
        # # NDVI_county_range= NDVI_P_range.where(mask==ID, np.nan)
        # # NDVI_county_mean_range=NDVI_county_range.mean(dim=('latitude', 'longitude'))        
        # # ## NDVI crop 
        # # NDVI_county_crop= NDVI_P_crop.where(mask==ID, np.nan)
        # # NDVI_county_mean_crop=NDVI_county_crop.mean(dim=('latitude', 'longitude'))    
        
        
                
                
        # # plt.close()   
    



        # #%% PEV plot( paper supplement)
        # Fval_merged_crop=Fval_merged#.where(app_mask.mask==1, np.nan)
        # #Fval_merged_crop= Fval_merged.where(mask==ID, np.nan) 
        
        # fig=plt.figure(figsize=(50,10)) # W,H

        # # if validation_length== 'seasonal': 
        # #     plt.suptitle('Economic value -- season: %s, severity threshold=%s'%(TOI, (str(selected_quantile*100)[:-2]+' %')), size=40, fontweight='bold')# (W,H)
        # # else: 
        # #     plt.suptitle('Economic value -- month: %s, severity threshold=%s'%(month_names[int(TOI)], (str(selected_quantile*100)[:-2]+' %')), size=40, fontweight='bold')# (W,H)
    
    
        # ## Axis for economic value plot 
        # gs=fig.add_gridspec(5,21,wspace=1,hspace=1)
        # ax1=fig.add_subplot(gs[0:3,0:3]) # Y,X
        # ax2=fig.add_subplot(gs[0:3,3:6],sharey=ax1)
        # ax3=fig.add_subplot(gs[0:3,6:9],sharey=ax1)
        # ax4=fig.add_subplot(gs[0:3,9:12],sharey=ax1)
        # ax5=fig.add_subplot(gs[0:3,12:15],sharey=ax1)  
        # ax6=fig.add_subplot(gs[0:3,15:18],sharey=ax1)         
        # ax7=fig.add_subplot(gs[0:3,18:21])

        # plt.setp(ax2.get_yticklabels(), visible=False)
        # plt.setp(ax3.get_yticklabels(), visible=False)
        # plt.setp(ax4.get_yticklabels(), visible=False)
        # plt.setp(ax5.get_yticklabels(), visible=False)
        # plt.setp(ax6.get_yticklabels(), visible=False)
        # plt.setp(ax7.get_yticklabels(), visible=False)        

    
    
        # label_x= 'C/L ratio'
        # label_y= 'PEV' #V$_{ECMWFseas5}$
        # label_x_size=60
        # label_y_size= 60
        # label_fontsize=50
        # x_ticks=[0.2, 0.6, 1]
        # y_ticks=[0.2, 0.6]
        # tick_size= 50
        # title_size=60
        # linewidth=12
        # colors=['darkblue','red','darkgreen']
        # ######################################################################### ax1 
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=0)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax1.plot(C_L, Fval,color=colors[p_thresholds.index(p)], label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)


        

        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax1.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax1.set_xlabel(label_x, size=label_x_size)
        # ax1.set_ylabel(label_y, size=label_y_size, weight = 'bold')  
        # #ax1.legend(fontsize=label_fontsize)    
        # ax1.set_xlim([0, 1])
        # ax1.set_ylim([0, 0.6])
        
        # ax1.set_xticks(x_ticks)
        # ax1.set_yticks(y_ticks)
        # ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax1.tick_params(axis='both', which='minor', labelsize=tick_size)
        
        

        # ######################################################################### ax2
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=1)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax2.plot(C_L, Fval, color=colors[p_thresholds.index(p)], label='Probability threshold= %s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
            
            
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax2.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax2.set_xlabel(label_x, size=label_x_size)
        # #ax2.set_ylabel(label_y, size=20)  
        # #ax2.legend(fontsize=label_fontsize)    
        # ax2.set_xlim([0, 1])
        # ax2.set_ylim([0, .6])     
        # ax2.set_xticks(x_ticks)
        # ax2.set_yticks(y_ticks)
        # ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax2.tick_params(axis='both', which='minor', labelsize=tick_size)        
        # ######################################################################### ax3
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=2)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax3.plot(C_L, Fval,color=colors[p_thresholds.index(p)], label='%s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
            
            
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax3.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax3.set_xlabel(label_x, size=label_x_size)
        # #ax3.set_ylabel(label_y, size=20)  
        # #ax3.legend(fontsize=label_fontsize)  
        # ax3.set_xlim([0, 1])
        # ax3.set_ylim([0, .6])    
        # ax3.set_xticks(x_ticks)
        # ax3.set_yticks(y_ticks)
        # ax3.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax3.tick_params(axis='both', which='minor', labelsize=tick_size)
        # ######################################################################### ax4
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=3)
        #     C_L=Fval_lead.C_L.values
       
        #     Fval=Fval_lead.sel(p=p).Fval.values
       
        #     ### plot 
        #     ax4.plot(C_L, Fval, color=colors[p_thresholds.index(p)], label='%s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
           
           
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax4.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # ax4.set_xlabel(label_x, size=label_x_size,weight = 'bold')
        # #ax4.set_ylabel(label_y, size=label_y_size)  
        
        # ax4.legend(title='Probability threshold',title_fontsize=label_fontsize, fontsize=label_fontsize, loc='lower center', bbox_to_anchor=(0.5, -1.1), ncol=3, fancybox=True, shadow=True)
        # ax4.set_xlim([0, 1])
        # ax4.set_ylim([0, .6])  
        # ax4.set_xticks(x_ticks)
        # ax4.set_yticks(y_ticks)
        # ax4.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax4.tick_params(axis='both', which='minor', labelsize=tick_size)
        # ######################################################################### ax5
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=4)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax5.plot(C_L, Fval, color=colors[p_thresholds.index(p)], label='%s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
            
            
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax5.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax5.set_xlabel(label_x, size=label_x_size)
        # #ax5.set_ylabel(label_y, size=20)  
        # #ax5.legend(fontsize=label_fontsize)    
        # ax5.set_xlim([0, 1])
        # ax5.set_ylim([0, .6])     
        # ax5.set_xticks(x_ticks)
        # ax5.set_yticks(y_ticks)
        # ax5.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax5.tick_params(axis='both', which='minor', labelsize=tick_size)
        # ######################################################################### ax6
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=5)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax6.plot(C_L, Fval, color=colors[p_thresholds.index(p)], label='%s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
            
            
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax6.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax6.set_xlabel(label_x, size=label_x_size)
        # #ax6.set_ylabel(label_y, size=20)  
        # #ax6.legend(fontsize=label_fontsize)    
        # ax6.set_xlim([0, 1])
        # ax6.set_ylim([0, .6])   
        # ax6.set_xticks(x_ticks)
        # ax6.set_yticks(y_ticks)
        # ax6.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax6.tick_params(axis='both', which='minor', labelsize=tick_size)
        # ######################################################################### ax7
        # for p in p_thresholds: 
        #     Fval_region=Fval_merged_crop.mean(dim=('latitude','longitude'))
        #     Fval_lead=Fval_region.isel(lead=6)
        #     C_L=Fval_lead.C_L.values
        
        #     Fval=Fval_lead.sel(p=p).Fval.values
        
        #     ### plot 
        #     ax7.plot(C_L, Fval, color=colors[p_thresholds.index(p)], label='%s'%(str(p*100)[:-2]) +'%', linewidth=linewidth)
            
            
        # lead=str(int(round((float(Fval_lead.lead.values)/(2592000000000000)),0)))
        # ax7.set_title('lead=%s'%(lead), size=title_size)   
        # ## axis labels 
        # #ax7.set_xlabel(label_x, size=label_x_size)
        # #ax7.set_ylabel(label_y, size=label_y_size)  
        # #ax7.legend(fontsize=label_fontsize)    
        # ax7.set_xlim([0, 1])
        # ax7.set_ylim([0, .6])      
        # ax7.set_xticks(x_ticks)
        # ax7.set_yticks(y_ticks)
        # ax7.tick_params(axis='both', which='major', labelsize=tick_size)
        # ax7.tick_params(axis='both', which='minor', labelsize=tick_size)
        
        # plt.show()

        
        # fig.savefig((os.path.join(path, 'plots\\python\\PEV_SUP%s.pdf'))%(TOI), bbox_inches='tight')

        # plt.close()


#%% Issue with xr.merge 
# Fval_merged.where((Fval_merged>0) | (Fval_merged!=np.nan), 0) ## xr.where(condition, outcome if condition is false) Returns dataset
# ## xr.where operation on a dataarray 
# test_array=Fval_merged['Fval'].where((Fval_merged['Fval']>0) & (Fval_merged['Fval']!=np.nan),Fval_merged['Fval'],0) ## xr.where(condition, outcome if condition is true, outcome if condition is false) Returns array. 


