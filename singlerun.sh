#! /bin/zsh

TH=1
name="vs0.75"
descp=(U*)
d=$(pwd)
start=0
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

c=$start
for f in ${~descp}
do
	echo $c
	#rm ${f}/log*
	full=${d}/$f
	#echo $full
	cp ~/Desktop/github/dpt/*py $full
	cd $f
	taskset -c $c python submittable.py &> log$c &
	cd $d
	let c=c+1
done
