#!/bin/bash
f="$1"
if [[ ${#f}  == 0 ]]; then
	echo "ERROR NO INPUT FILENAME"
	exit -1
fi

folder="$(dirname $f)"
cd $folder
ar x $(basename $f)
cd -
rm $f
