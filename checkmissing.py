import subprocess
from glob import glob
from os import getcwd, path
import numpy as np


def searchkey(needle, haystack : str, default_key = None, whichseg = -1, default_value = '', func = lambda x: x)  : 

    haystack = haystack.strip('/')
    haystack = haystack.split('/')[whichseg]
    strs = haystack.split('_')

    try:
        for val in strs:
            if val[:len(needle)] == needle:
            #and not val[len(needle)].isalpha() :
                return func(val[len(needle):])
            
        if default_key is not None :
            return func(searchkey(default_key, haystack))
        
        return func(default_value)
    except:
        print(needle)
        print("Have you checked needle and haystack?")

# check if the jobs are still running by checking the slurm output, if not, print out the terminated ones
def check_missing(default = "U*"):
    output = subprocess.run(['squeue', '-u', 'knl20'], capture_output=True)
    output = output.stdout.splitlines()

    curjobs  = set()
    
    for line in output[1:]:

        vals = line.split()
        jobnum = int(vals[0])
        curjobs.add(jobnum)

    
    pwd = getcwd()
    fs = glob( f'{pwd}/{default}/sl*')


    terminatedlog = set()
    terminateddir = set()
    for f in fs:
        
        *parent, num = f.split('/')

        parent = '/'.join(parent)
        if path.isfile(f'{parent}/CALC_FIN'):
            continue

        prefix = 'slurm-'
        suffix = '.out'
        num = num.strip(prefix)
        num = num.strip(suffix)

        num = int(num)
        
        if num not in curjobs:
            terminatedlog.add( f)
            terminateddir.add( parent)


    print("terminated log: ", terminatedlog)
    np.savetxt( f'{pwd}/TERMINATEDDIR', list(terminateddir), fmt = '%s')


# we strictly need n1 to run to full
def check_TOL(default = "U*"):

    fs = glob( f'{getcwd()}/{default}')

    for f in fs:

        bf = glob( f'{f}/bs_*')
        bfs = sorted(bf, key = lambda x: int(x.split('_')[-1][len('repeat'):]))

        tfin = searchkey('tfin', f, func = float)
        
        try:
            secondlast = bfs[-2]
            lo = np.loadtxt(f'{secondlast}/lo')
            hi = np.loadtxt(f'{secondlast}/hi')

            mid = (lo + hi) / 2
        except:
            continue

        last = bfs[-1]
        n1 = np.loadtxt(f'{last}/n1')
        time = np.loadtxt(f'{last}/times')[-1]
        n1last = np.mean(n1[-8:])
        if time == tfin and np.abs(n1last - mid) < 1e-5:

            print(f)
            np.savetxt(f'{f}/TOL_REACHED', [])

if __name__ == '__main__':
    check_missing(default = "*dim1024*")
    #check_TOL()