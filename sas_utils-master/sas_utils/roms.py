import datetime, math, urllib, os, pdb, itertools
from scipy.interpolate import griddata
from tqdm import tqdm
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




def getROMSData(datafile_path, feature):
  #Load a single Roms Feature as a scalar field, return the field and its bounds
  #Note that the field is a masked array, so all locations within the bounds are not guaranteed to be valid


  if 'txla' in datafile_path:
    #ROMS Data is from Texas - Lousisiana Dataset
    return loadTXLAROMSData(datafile_path, feature)
    # scalar_field, scalar_lat, scalar_lon, roms_t = loadTXLAROMSData(datafile_path, feature)
    # current_u, u_lat, u_lon, _ = loadTXLAROMSData(datafile_path, "current_u")
    # current_v, v_lat, v_lon, _ = loadTXLAROMSData(datafile_path, "current_v")


    # [scalar_field, current_u, current_v], y_ticks, x_ticks, t_ticks, lon_ticks, lat_ticks = reshapeROMS([scalar_field, current_u, current_v], roms_lat, roms_lon, t_ticks, bounds, resolution, xlen, ylen)


  # elif 'ocean_his' in datafile_path:
  #  #ROMS Data is from Oregon Dataset
  #  return loadOregonROMSData(datafile_path, feature, bounds, resolution)

  # elif 'ca300m' in datafile_path:
  #  #ROMS Data is from Monterey Dataset
  #  return loadMontereyROMSData(datafile_path, feature, bounds, resolution)



# def loadOregonROMSData(datafile_path, feature="temperature", bounds=None, resolution=None):
#  # Returns Ocean Surface Temperature
#  roms_dataset = nc.Dataset(datafile_path)

#  if feature == 'temp' or feature == 'temperature':
#    lat = roms_dataset['lat_rho'][:,0]
#    lon = roms_dataset['lon_rho'][0,:]

#    scalar_field = roms_dataset['temp'][0,39,:,:]

#  elif feature == 'current_u' or feature == "u":
#    lat = roms_dataset['lat_u'][:,0]
#    lon = roms_dataset['lon_u'][0,:]

#    scalar_field = roms_dataset['u'][0,39,:,:]   

#  elif feature == 'current_v' or feature == "v":
#    lat = roms_dataset['lat_v'][:,0]
#    lon = roms_dataset['lon_v'][0,:]

#    scalar_field = roms_dataset['v'][0,39,:,:]  


#  if bounds is not None and resolution is not None:
#    return reshapeROMS(scalar_field, lat, lon, bounds, resolution)
#  else:
#    return scalar_field, lat, lon



# def loadMontereyROMSData(datafile_path, feature="temperature", bounds=None, resolution=None):
#  roms_dataset = nc.Dataset(datafile_path)

  
#  if feature == "temp" or feature == "temperature":
#    scalar_field, lat, lon = slice(roms_dataset, 'temp')

#  elif feature == "salinity" or feature == "salt":
#    scalar_field, lat, lon = slice(roms_dataset, 'salt')

#  elif feature == "current_u" or feature == "u":
#    scalar_field, lat, lon = slice(roms_dataset, 'u')

#  elif feature == "current_v" or feature == "v":
#    scalar_field, lat, lon = slice(roms_dataset, 'v')

#  if bounds is not None and resolution is not None:
#    return reshapeROMS(scalar_field, lat, lon, bounds, resolution)
#  else:
#    return scalar_field, lat, lon



def loadTXLAROMSData(datafile_path, feature='temperature'):
  roms_dataset = nc.Dataset(datafile_path)

  if feature == 'temp' or feature == 'temperature':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['temp'][:,0,:,:]

  elif feature == 'salt' or feature == 'salinity':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['salt'][:,0,:,:]

  elif feature == 'current_u' or feature == "u":
    lat = roms_dataset['lat_u'][:]
    lon = roms_dataset['lon_u'][:]
    times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['u'][:,0,:,:]   

  elif feature == 'current_v' or feature == "v":
    lat = roms_dataset['lat_v'][:]
    lon = roms_dataset['lon_v'][:]
    times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['v'][:,0,:,:]  

  return scalar_field, lat, lon, times

def reshapeROMS(roms_field, roms_lat, roms_lon, bounds, output_shape):
  n_bound   = bounds[0]
  s_bound   = bounds[1]
  e_bound   = bounds[2]
  w_bound   = bounds[3]

  lonlon,latlat = np.mgrid[w_bound:e_bound:output_shape[0]*1j, s_bound:n_bound:output_shape[1]*1j]

  if roms_lat.ndim == 1 and roms_lon.ndim == 1:
    roms_lat, roms_lon = np.meshgrid(roms_lat, roms_lon)

  lat_coords = roms_lat.flatten()
  lon_coords = roms_lon.flatten()

  pts = np.vstack((lon_coords, lat_coords)).transpose()
  
  reshaped_field = np.empty(output_shape)
  for t_idx in tqdm(range(output_shape[2])):
    data = roms_field[t_idx].data.flatten()
    zz = griddata(pts, data, (lonlon,latlat), fill_value=9999.)
    reshaped_field[:,:,t_idx] = zz

  reshaped_field = np.ma.masked_greater(reshaped_field, 1.1*np.max(roms_field))

  return reshaped_field
