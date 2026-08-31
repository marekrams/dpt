import numpy as np
from yastn.operators import SpinlessFermions
import yastn.tn.mps as mps
from glob import glob
import os
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from functools import reduce

def recal_ent(fs):

    for i, f in enumerate(fs):

        print(i)
        fstr = f'{f}/iter_repeat0'

        if not os.path.exists(f'{fstr}/SvN'):
            print(f"dir {f} does not have SvN!")
            continue 

        SvN = np.loadtxt( f'{fstr}/SvN')

        effE = np.log( np.power( np.sum( np.exp( 3 * SvN), axis = 1) / ( SvN.shape[-1] ), 1/3))
        np.savetxt( f'{fstr}/Seff', effE)
        


def zipfiles(fs):

    datas = [
        'Seff',
        'MaxEnt',
        'n1',
        'times',
        'sites'
    ]

    archive_anchor = Path( os.getcwd())

    with ZipFile("output.zip", "w", compression=ZIP_DEFLATED) as archive:
        for i, f in enumerate(fs):

            fstr = f'{f}/iter_repeat0'
            print('compress ', i)
            for data in datas:

                df = Path(f'{fstr}/{data}')
                archive.write(df, arcname=df.relative_to(archive_anchor))

            logs = glob(f'{f}/sl*')
            for log in logs:

                dl = Path(log)
                archive.write(dl, arcname=dl.relative_to(archive_anchor))

            dp = Path(f'{f}/dptpara.json')
            archive.write(dp, arcname=dp.relative_to(archive_anchor))

if __name__ == '__main__':

    TOP = os.getcwd()


    allf = [glob(f'{TOP}/ProdAug5/U*'),
            glob(f'{TOP}/ProdAug6/U*'),
              glob(f'{TOP}/ProdAug10/U*'),
              glob(f'{TOP}/Aug1log/U*'),
              glob(f'{TOP}/Aug13logcompletion/U*'), 
              glob(f'{TOP}/ProdAug20spatial/U*')]
    print( [ len(f) for f in allf])

    fs = reduce( lambda x, y: x + y, allf)

    recal_ent(fs)
    zipfiles(fs)