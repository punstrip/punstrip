#!/bin/bash
BINS_PATH="/root/elf_bins"
DBG_BINS_PATH="/root/dbgsym_bins"

function merge_debug_info(){
	dbg_info="$1"
	bin="$2"

	tmp_dir="/tmp/unstrip"
	mkdir -p "$tmp_dir"

	cp "$bin" "$tmp_dir"
	cp "$dbg_info" "$tmp_dir"

	bbin="$(basename $bin)"
	bdbg_info="$(basename $dbg_info)"

	echo "Unstripping $bbin"

	eu-unstrip "$tmp_dir/$bbin" "$tmp_dir/$bdbg_info"
	cp "$tmp_dir/$bdbg_info" "$DBG_BINS_PATH/$bbin"

	rm "$tmp_dir/$bbin" "$tmp_dir/$bdbg_info"
}


mkdir -p "$BINS_PATH"
mkdir -p "$DBG_BINS_PATH"

pushd .
cd deb

for f in $(find . -maxdepth 1 -type d -not -name '*.deb' -not -name '*-dbgsym' -not -name '.')
do
	echo "Using $f"
	for j in $(find $f -type f)
	do
		echo "$j"
		if [ ! -z "$(file $j | grep ELF)" ]
		then
			build_id="$(file "$j" | cut -f 2 -d '=' | cut -f 1 -d ',')"
			echo "Build ID: $build_id"
			#echo "Searching $f-dbgsym/"
			dbg_info=$(find "$f-dbgsym" -name "*${build_id:2}.debug" 2>/dev/null)
			echo "DBG_INFO: $dbg_info"

			if [ ! -z "$dbg_info" ]; then
				merge_debug_info "$dbg_info" "$j"
			fi

		elif [ -z "$(file $j | grep 'ar archive')" ]; then


		fi
	done
done

popd
