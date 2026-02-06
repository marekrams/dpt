#! /bin/bash

c=$(date +%s)
TH=1
#TH=1
d=$(pwd)
for f in $(ls -d U*)
do
	let c=c+1
	full=${d}/$f
	echo $full
	cp ~/Desktop/github/dpt/*py $full
	tmux new-session -d -s "$c" -c "$full" \
	 "OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH BLIS_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
	python submittable.py &> log"
	echo $c
done
