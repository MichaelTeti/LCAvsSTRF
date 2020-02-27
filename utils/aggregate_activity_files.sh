while getopts ":f:t:k:" arg; do
	if [ $arg == f ]; then
		from=$OPTARG;
	elif [ $arg == t ]; then
		to=$OPTARG;
	elif [ $arg == k ]; then
		key=$OPTARG;
	fi
done


for i in $from/*; do
    for j in $i/$key; do
        d=$(dirname $j)
        d=$(basename $d)
        cp $j $to$d".pvp"
    done;
done
