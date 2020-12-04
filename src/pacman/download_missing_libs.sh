#!/bin/bash
LIB="$1"
OUT_FILE="/root/LIBS/packages"

if [ -z "$LIB" ]; then
	echo "Error, no library argument passed!"
	exit
fi

echo "Finding packages containing $LIB..."

packages=$(apt-file search "$LIB" | cut -f 1 -d ':' | sort | uniq)

for p in $packages; do
	echo $p >> $OUT_FILE
done
