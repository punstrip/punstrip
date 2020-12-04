#!/bin/bash
## Extract all ELF objects from SRC and moves to DST
SRC_DIR="$1"
DST_DIR="$2"

source /usr/bin/env_parallel.bash

#exit on error
set -e

if [[ $# != 2 ]]; then
	echo "Usage: ./$0 SRC DEST"
	exit
fi

if [ ! -d "$SRC_DIR" ]; then
	echo "Error, please supply a source directory"
	exit
fi
if [ ! -d "$DST_DIR" ]; then
	echo "Error, please supply a destination directory"
	exit
fi

echo "Extracting ELF objects from $SRC_DIR to $DST_DIR"

echo "Finding files..."
#files=$(find "$SRC_DIR" -type f)

check_file(){
	f="$1"
	folder="$(dirname $f)"
	fname="$(basename $f)"
	relsrc="$(basename $SRC_DIR)"
	newfolder="${folder##./$relsrc}"

	##check if already copied
	if [ -f "$DST_DIR/${newfolder:1}/$fname" ]; then
		echo "[-] Already processes $f"
		return
	fi
	if [ ! -z "$(file $f | grep -P 'ELF|ar archive')" ]
	then
		mkdir -p "$DST_DIR/${newfolder:1}"

		echo "[+] Found $f"
		ln "$f" "$DST_DIR/${newfolder:1}"
	fi

}

find "$SRC_DIR" -type f | env_parallel check_file

#echo $files | parallel check_file
#for f in $files; do
#	check_file $f
#done
