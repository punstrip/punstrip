#!/bin/bash

PKG_DIR="$1"
echo "Extract all packages found in $PKG_DIR"

function extract_deb(){
	fname="$1"
	dir="$(dpkg -f $fname Package)"
    if [ ! -d "$dir" ]; then
        echo "Extracting $dir..."
        dpkg -x "$f" "$dir"
    fi
}

pushd .
cd deb

for f in $(find . -maxdepth 1 -type f -iname '*.deb' )
do
	extract_deb "$f"
done

popd
