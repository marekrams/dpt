#! /bin/zsh

#TH=4
TH=1

c=$(pwd)
tmux new-session -d -s "test dpt" -c $c \
"OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH BLIS_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
perf stat -e instructions,cycles,stalled-cycles-frontend,stalled-cycles-backend,cache-misses python submittable.py &> log"
