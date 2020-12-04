#!/bin/bash
#############
#### To be ran as the argument of find | GNU parallel
#############

if [[ ${#1}  == 0 ]]; then
	echo "ERROR NO INPUT FILENAME"
	exit -1
fi

f="$1"

OUT_BIN_PATH="../elf_bins"
BIN_NAME="$(basename $f)"
BIN_DIR="$(dirname $f)"

if [ -f "$OUT_BIN_PATH/${BIN_DIR:1}/$BIN_NAME" ]; then
	echo "Skipping, $f exists in $OUT_BIN_PATH"
	exit
fi

if [ ! -z "$(file $f | grep -P 'ELF|ar archive')" ]
then
	echo "Copying $f"
	mkdir -p "$OUT_BIN_PATH/${BIN_DIR:1}"
	cp "$f" "$OUT_BIN_PATH/${BIN_DIR:1}"
fi
