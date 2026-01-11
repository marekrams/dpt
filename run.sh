#! /bin/zsh

c=0
d=$(pwd)
for f in $(ls -d U*)
do
	let c=c+1
	full=${d}/$f
	echo $full
	cp ~/Desktop/github/dpt/*py $full
	tmux new-session -d -s "$c" -c "$full"  "python submittable.py &> log"
	echo $c
done
