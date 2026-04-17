import subprocess
from glob import glob
from os import getcwd


def check_missing():
    output = subprocess.run(['squeue', '-u', 'knl20'], capture_output=True)
    output = output.stdout.splitlines()

    curjobs  = set()
    
    for line in output[1:]:

        vals = line.split()
        jobnum = int(vals[0])
        curjobs.add(jobnum)

    
    pwd = getcwd()
    fs = glob( f'{pwd}/U*/sl*')


    terminated = set()
    for f in fs:
        
        num = f.split('/')[-1]

        prefix = 'slurm-'
        suffix = '.out'
        num = num[len(prefix):]
        num = num[:-len(suffix)]

        num = int(num)
        
        if num not in curjobs:
            terminated.add( f)


    print(terminated)

if __name__ == '__main__':
    check_missing()