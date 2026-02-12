#! /bin/bash

c=$(date +%s)
TH=1
name="vs0.75"
descp="U*"
#TH=1
d=$(pwd)
for f in $(ls -d ${descp})
do
	let c=c+1
	full=${d}/$f
	echo $full
	cp ~/Desktop/github/dpt/*py $full
	tmux new-session -d -s "$c TH$TH $name" -c "$full" \
	 "OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH BLIS_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH \
	python submittable.py &> log"
	echo $c
done
