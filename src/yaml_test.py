import oyaml as yaml
import os

def main():
    with open(os.path.expandvars('cfg/robots.yaml'),'rb') as f:
        yaml_mission = yaml.load(f.read())

    plan_robots = ['glider1', 'rose1']

    for key,value in [(k,v) for k,v in yaml_mission.items() if k in plan_robots]:
        # if key in plan_robots:
        print(key)
        print(value['vel'])
        print(value)

if __name__ =="__main__":
    main()
