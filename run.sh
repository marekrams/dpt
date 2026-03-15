#! /bin/zsh

c=$(date +%s)
TH=1
name="vs0.75"
descp=(U*)
d=$(pwd)
for f in ${~descp}
do
	let c=c+1
	rm ${f}/log*
	full=${d}/$f
	echo $full
	cp ~/Desktop/github/dpt/*py $full
	tmux new-session -d -s "$c TH$TH $name" -c "$full" \
  "zsh -lc '
    export OMP_NUM_THREADS=$TH
    export MKL_NUM_THREADS=$TH
    export OPENBLAS_NUM_THREADS=$TH
    export BLIS_NUM_THREADS=$TH
    export VECLIB_MAXIMUM_THREADS=$TH
    export NUMEXPR_NUM_THREADS=$TH
    export MKL_DYNAMIC=FALSE
    export OMP_DYNAMIC=FALSE

    python submittable.py &> log$c
    rc=\$?
    print -r -- \"EXIT=\$rc at \$(date -Is)\" >> log$c
    exit \$rc
  '"
done
