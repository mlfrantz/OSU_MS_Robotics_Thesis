#!/home/mlfrantz/miniconda2/bin/python3.6

import oyaml as yaml
import sys

filename = '/home/mlfrantz/Documents/MIP_Research/mip_research/cfg/sim_test.yaml'

with open(filename,'rb') as f:
    yaml_sim = yaml.load(f.read())

yaml_sim['sim_world']['center_latitude'] = float(sys.argv[1])
yaml_sim['sim_world']['center_longitude'] = float(sys.argv[2])

with open(filename,'w') as outfile:
    yaml.dump(yaml_sim, outfile, default_flow_style=False)
