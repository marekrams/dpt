#! /bin/zsh

#TH=4
TH=1

c=$(pwd)
tmux new-session -d -s "test dpt" -c $c \
"OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH BLIS_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
python submittable.py &> log"
