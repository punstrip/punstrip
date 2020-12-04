#!/bin/bash
DBG_BINS_PATH="/root/dbg_elf_bins"

f="$1"
if [[ ${#f} == 0 ]]; then
	echo "ERROR, no file name given to script!"
	exit
fi

if [ -z "$(file $f | grep 'not stripped')" ]; then
	echo "ERROR, $f is stripped!"
	exit
fi

bbin=$(basename $f)
pbin=$(dirname $f | awk -F 'elf_bins/' '{print $2}')

if [ -f "$DBG_BINS_PATH/$pbin/$bbin" ]; then
	echo "Skipping, file already exists in $DBG_BINS_PATH"
	exit
fi

#echo "Binary: $bbin"
#echo "Making $DBG_BINS_PATH/$pbin"

if [ ! -d "$DBG_BINS_PATH/$pbin" ]; then
	mkdir -p "$DBG_BINS_PATH/$pbin"
fi

cp "$f" "$DBG_BINS_PATH/$pbin/$bbin"
