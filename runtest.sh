#! /bin/zsh

TH=1
#TH=1

export OMP_NUM_THREADS=$TH
export MKL_NUM_THREADS=$TH
export OPENBLAS_NUM_THREADS=$TH
export BLIS_NUM_THREADS=$TH
export VECLIB_MAXIMUM_THREADS=$TH
export NUMEXPR_NUM_THREADS=$TH 
python submittable.py
