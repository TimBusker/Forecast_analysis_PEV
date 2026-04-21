# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:04:50 2020

@author: tbr910

Needs to be run in the ML_env environment, otherwise it will not work.
"""
#!/usr/bin/env python
import sys

import os
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns
import pandas as pd


from ecmwfapi import ECMWFService
from ecmwfapi import ECMWFDataServer
#import cfgrib
#import pygrib
#from netCDF4 import Dataset
#import xarray as xr 

#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.basemap import shiftgrid
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

#####################################################################################################################################################################
###################################################################### MARS ARCHIVE ##########################################################################
os.chdir('/scistor/ivm/tbr910/ECMWF') ## Dir has to be tbr910, for some weird reason
################################# SEAS forecast ############################################### 


# server = ECMWFService("mars")



# def retrieve_mars_data():
    

#     date_list=pd.date_range('2012-01-01','2013-01-01', 
#               freq='MS').strftime("%Y-%m-%d").tolist()#YS
#     date_range= []
#     for i in range(len(date_list)-1): 
#         # if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1]:
#             month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
#             date_range.append(month_range)
#     #times = ['00', '12']
  
#     for date in date_range:
#         target = 'C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF/mars_files_used/ecmwf_ens_%s.grb' % (date[:10])
#         print ('REQUEST ACTIVE FOR: %s' % (target))
#         mars_ens_request(date, target)
                          
# def mars_ens_request (date, target):              
#     server.execute(
#         {
#         "class": "od",
#         "date": date,
#         "expver": "1",
#         "levtype": "sfc",
#         "param": "228.128",
#         "step": "6/TO/36/BY/6", 
#         "stream": "enfo",
#         'number':'1/to/50',
#         "time":  "00:00:00/12:00:00",
#         "type": "pf",
#         "area": "54.109311/2.064148/50.794060/7.788420",
#         "grid": "F640"
#         },
#         target)

# retrieve_mars_data()





################################# EFI/SOT ###############################################

# #os.chdir('C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF') ## Dir has to be tbr910, for some weird reason
# server = ECMWFService("mars")
# variable='efi' # sot or efi 
# def retrieve_mars_data():
    
#     month_list= pd.date_range('2022-12-01', '2023-12-01', freq='MS').strftime("%Y-%m").tolist()



#     # take first and last day of each month from month_list and make a string 2010-09-01/to/2010-09-30 
#     date_range= []
#     for i in range(len(month_list)-1): 
#         first_month= '%s-01' % (month_list[i])
#         # I want last month to be month_list[i+1] -1 day, but I have to convert it to datetime object first
#         last_month= datetime.strptime(month_list[i+1], '%Y-%m') - pd.Timedelta(days=1)
#         last_month= last_month.strftime("%Y-%m-%d")
#         month_range= '%s/to/%s' % (first_month, last_month)
#         date_range.append(month_range)



#     times = ['00:00:00']
    
#     # invert date_range items (o.e. last one becomes first one etc.)
#     date_range= date_range[::-1]

#     for date in date_range:
#         for time in times:
#             target = os.getcwd()+'/files_%s/%s_europe_%s_%s.grb' % (variable, variable, date[:-14], time[:2])
#             print ('REQUEST ACTIVE FOR: %s' % (target))
#             print (date, time)
#             mars_ens_request(date,time, target)
                                
# def mars_ens_request (date,time, target):              
#     server.execute(
#         {
#         "class": "od",
#         "date": date,
#         "expver": "1",
#         "levtype": "sfc",
#         "param": "228.132",
#         "step": ' 0-24/24-48/48-72/72-96/96-120',
#         "stream": "enfo",
#         "time":  time,
#         "type": variable,   
#         "number": "90",
#         "grid": 'F640', #O1280
#         "area": "Europe", #Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
#         },
#         target)

# retrieve_mars_data()



# ################################# Medium-term Rainfall forecast for whole europe ###############################################

os.chdir('/scistor/ivm/tbr910/ECMWF') ## Dir has to be tbr910, for some weird reason
server = ECMWFService("mars")

def retrieve_mars_data():
    
    # make list of dates between 2000 and 2023 (daily)
    date_list=pd.date_range('2007-01-01','2023-01-01', # 2007 start of total precip
                freq='D').strftime("%Y-%m-%d").tolist()#YS
    print(date_list)
    #date_list=['2021-07-15'] # '2021-07-09', '2021-07-10', '2021-07-11', '2021-07-12', '2021-07-13', '2021-07-14', 
    #date_range= []
    # for i in range(len(date_list)-1): 
    #     # if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1]:
    #         month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
    #         date_range.append(month_range)
    times = ['00'] # initialization time 
  
    for date in date_list:
        for time in times:
            target = os.getcwd()+'/files_europe/ecmwf_europe_%s_%s.grb' % (date[:10], time[:2])
            print ('REQUEST ACTIVE FOR: %s' % (target))
            print (date, time)
            mars_ens_request(date,time, target)
                          
def mars_ens_request (date,time, target):              
    server.execute(
        {
        "class": "od",
        "date": date,
        "expver": "1",
        "levtype": "sfc",
        "param": "228.128",
        "step": '0/TO/240/BY/24',
        "stream": "enfo",
        'number':'1/to/50',
        "time":  time,
        "type": "pf",   
        "grid": 'F640', #O1280
        "area": "Europe", #Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
        },
        target)

retrieve_mars_data()




# def tigge_pf_request (date, target):              
#     server.retrieve({
        
#         "class": "ti",
#         "dataset": "tigge",
#         "date": date,
#         "expver": "prod",
#         "grid": "F640", 
#         "levtype": "sfc",
#         "number": "1/to/50",#/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50
#         "origin": "ecmf",
#         "param": "228228",
#         "step": "6/TO/36/BY/6", # 0 is time step zero, so zero accumulated precipitation. Accumulated precipitation (tp) is compared to initialization time (i.e. step 36 is accumulated tp from ini to step 36).
#         "time": '00/12',
#         "area": " 54.109311/2.064148/50.794060/7.788420",    # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
#         "type": "pf", # perturbed forecast fp= forecast probability , cf=control forecast,
#         "target": target,
#     })


# retrieve,
# class=od,
# date=['2021-07-09', '2021-07-10', '2021-07-11', '2021-07-12', '2021-07-13', '2021-07-14', '2021,07-15']
# expver=1,
# levtype=sfc,
# param=228.128,
# step=
# stream=oper,
# time=00:00:00, 12:00:00
# type=fc,
# target="output"

# Bounding box in EPSG = 4326: 
# North: 50.95
# East: 6.15
# South: 50.60
# West: 5.65

# #####################################################################################################################################################################
# ###################################################################### TIGGE ARCHIVE ##########################################################################
# server = ECMWFDataServer()

# def retrieve_tigge_data():
    
# # old request setting
#     # date_list=pd.date_range('2010-01-01','2010-01-03', 
#     #           freq='D').strftime("%Y-%m-%d").tolist()#MS
#     # #date_range= []
#     # #forbidden_dates= ['2015-09-01','2014-03-01'] ## damaged tapes
#     # # for i in range(len(date_list)-1): 
#     # #     if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1] :
#     # #         month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
#     # #         date_range.append(month_range)

#     # forbidden_dates= ['2015-09-01','2014-03-01'] ## damaged tapes
#     # times = ['00', '12']
#     # for date in date_list:
#     #   if  date!=forbidden_dates[0] and date!=forbidden_dates[1]:  
#     #     for time in times:
#     #         target = 'C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF/tigge_files/12_h_runs/ecmwf_tigge_%s_%s.grb' % (date[:10], time)
#     #         print ('REQUEST ACTIVE FOR: %s' % (target))
#     #         tigge_pf_request(date,time, target)

# # new request setting
#     date_list=pd.date_range('2020-06-01','2020-07-01', 
#               freq='MS').strftime("%Y-%m-%d").tolist()#MS
#     date_range= []
#     for i in range(len(date_list)-1): 
#         # if  date_list[i]!=forbidden_dates[0] and date_list[i]!=forbidden_dates[1]:
#             month_range= '%s/to/%s' % (date_list[i], date_list[i+1]) 
#             date_range.append(month_range)
  
#     for date in date_range:
#         target = 'C:/Users/tbr910/Documents/Paper1/Analysis/ECMWF/tigge_files/ecmwf_tigge_%s.grib' % (date[:10])
#         print ('REQUEST ACTIVE FOR: %s' % (target))
#         tigge_pf_request(date, target)               
        
# def tigge_pf_request (date, target):              
#     server.retrieve({
        
#         "class": "ti",
#         "dataset": "tigge",
#         "date": date,
#         "expver": "prod",
#         "grid": "F640", 
#         "levtype": "sfc",
#         "number": "1/to/50",#/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50
#         "origin": "ecmf",
#         "param": "228228",
#         "step": "6/TO/36/BY/6", # 0 is time step zero, so zero accumulated precipitation. Accumulated precipitation (tp) is compared to initialization time (i.e. step 36 is accumulated tp from ini to step 36).
#         "time": '00/12',
#         "area": " 54.109311/2.064148/50.794060/7.788420",    # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
#         "type": "pf", # perturbed forecast fp= forecast probability , cf=control forecast,
#         "target": target,
#     })

# retrieve_tigge_data()