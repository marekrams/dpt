#! /bin/zsh

descp=(U*)
d=$(pwd)
rm jobs.txt
for f in ${~descp}
do
	if [[ ! -f ${f}/CALC_FIN ]]
	then

		full=${d}/$f
		echo $full
		cp ~/Desktop/github/dpt/*py $full
		echo "cd $full && python -u submittable.py" >> jobs.txt
	fi
done
