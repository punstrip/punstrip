#!/bin/bash
######
#### To be called with find elf_bins ELF_BINS -not -name '*.debug' | parallel ../merge_debug
######

OUTPUT_DIR="dbg_elf_libs"
SRC_DIR="elf_libs"
source /usr/bin/env_parallel.bash

f="$1"
if [[ ${#f} == 0 ]]; then
	echo "ERROR, no file name given to script!"
	exit
fi

if [ ! -z "$(file $f | grep 'with debug_info')" ]; then
	echo "ERROR, $f is already non stripped!"
	exit
fi

function merge_debug_info(){
	dbg_info="$1"
	bin="$2"

	dbg_info_fname="$(basename $dbg_info)"
	bin_fname="$(basename $bin)"

	tmp_dir=$(mktemp -d -p "/tmp/unstrip")

	ln "$bin" "$tmp_dir/$bin_fname"
	ln "$dbg_info" "$tmp_dir/$dbg_info_fname"

	bbin="$(basename $bin)"
	#pbin=$(dirname $bin | awk -F "$SRC_DIR" '{print $2}')
	dbin="$(dirname $bin)"
	pbin=${dbin#$SRC_DIR}

	bdbg_info="$(basename $dbg_info)"


	echo "Unstripping $bbin"

	echo "Making $OUTPUT_DIR/$pbin"
	mkdir -p "$OUTPUT_DIR/$pbin"

	eu-unstrip "$tmp_dir/$bbin" "$tmp_dir/$bdbg_info"
	echo "Copying $tmp_dir/$bdbg_info	to	$OUTPUT_DIR/$pbin/$bbin"
	ln "$tmp_dir/$bdbg_info" "$OUTPUT_DIR/$pbin/$bbin"

	rm "$tmp_dir/$bbin" "$tmp_dir/$bdbg_info"
}

bbin="$(basename $f)"
pbin=$(dirname $f | awk -F "$OUTPUT_DIR" '{print $2}')
if [ -f "$OUTPUT_DIR/$pbin/$bbin" ]; then
	echo "Skipping, file already in unstripped corpus"
	exit
else
	echo "unstripping $f, finding debug info..."
fi

#echo $f
build_id="$(file $f | cut -f 2 -d '=' | cut -f 1 -d ',')"
#echo "BUILD ID: $build_id"

debug_file_dir="$(echo $f | awk -F '/' '{print $1 "/" $2 "-dbgsym" }')"
if [ ! -d $debug_file_dir ]; then
	debug_file_dir="$(echo $f | awk -F '/' '{print $1 "/" $2 "-dbg" }')"
fi

if [ ! -d $debug_file_dir ]; then
	echo "ERROR, could not find debug info package directory"
	exit
fi 
#echo $debug_file_dir

#dbg_info=$(find . -name "*${build_id:4}.debug")
dbg_info=$(find $debug_file_dir -name "*${build_id:4}.debug")

if [[ ${#build_id} == 0 ]]; then
	echo "ERROR, failed to get build id!"
	exit
fi

if [ ! -z "$dbg_info" ]
then
	#echo "FOUND!"
	echo "$f :: $build_id :: $dbg_info"
	merge_debug_info "$dbg_info" "$f"
else
	echo "ERROR, No debug info for $f or build id $build_id"
fi
