# import numpy as np
#
# Np = 10
# steps = [range(0,55),range(0,55)]
# max_steps = max([len(s) for s in steps])
# field_delta = int(max_steps/Np)
# print(field_delta)
#
# t_step = 0
# k_step = 0
# field_time_steps = []
# for i in range(max_steps):
#     field_time_steps.append(t_step)
#     k_step += 1
#     if k_step == field_delta:
#         k_step = 0
#         t_step += 1
#
# print(field_time_steps, len(field_time_steps))

import sys, pdb, time, argparse, os
import oyaml as yaml
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
from sas_utils import World, Location
import imageio

with open(os.path.expandvars('cfg/sim.yaml'),'rb') as f:
    yaml_sim = yaml.load(f.read())

wd = World.roms(
    datafile_path=yaml_sim['roms_file'],
    xlen        = yaml_sim['sim_world']['width'],
    ylen        = yaml_sim['sim_world']['height'],
    center      = Location(xlon=yaml_sim['sim_world']['center_longitude'], ylat=yaml_sim['sim_world']['center_latitude']),
    feature     = yaml_sim['science_variable'],
    resolution  = (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution']),
    )

# This is the scalar_field in a static word.
# The '0' is the first time step and goes up to some max time
# field = np.copy(wd.scalar_field[:,:,0])
field = np.copy(wd.scalar_field)
field_resolution = (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution'])


for i in range(25):
    plt.imshow(wd.scalar_field[:,:,i].transpose(), interpolation='gaussian', cmap= 'gnuplot')
    plt.xticks(np.arange(0,len(wd.lon_ticks), (1/min(field_resolution))), np.around(wd.lon_ticks[0::int(1/min(field_resolution))], 2))
    plt.yticks(np.arange(0,len(wd.lat_ticks), (1/min(field_resolution))), np.around(wd.lat_ticks[0::int(1/min(field_resolution))], 2))
    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)
    plt.text(1.25, 0.5, str(yaml_sim['science_variable']),{'fontsize':20}, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False, transform=plt.gca().transAxes)

    file_string = 'mip_run_' + str(wd.t_ticks[i])[:-2]
    plt.savefig("/home/mlfrantz/Documents/MIP_Research/mip_research/movies/" + file_string)

images = []
print(sorted(os.listdir("/home/mlfrantz/Documents/MIP_Research/mip_research/movies/")))
for filename in sorted(os.listdir("/home/mlfrantz/Documents/MIP_Research/mip_research/movies/")):
    print(filename)
    images.append(imageio.imread("/home/mlfrantz/Documents/MIP_Research/mip_research/movies/" + filename))
imageio.mimsave("/home/mlfrantz/Documents/MIP_Research/mip_research/movies/movie.gif", images)
