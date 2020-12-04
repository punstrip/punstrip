#!/bin/bash

PACKAGES_SRC="$1"
PACKAGE_TYPE="dbgsym"
#PACKAGE_TYPE="dbg"

TYPE_LEN=$(( 1 + ${#PACKAGE_TYPE} ))

function check_package(){
	package="$1"
	res=$(find . -maxdepth 1 -iname "$package*.deb")
	if [ ! -z "$res" ]; then
		echo 1
	fi
	echo 0
}

function extract_deb(){
	fname="$1"
	dir="$(dpkg -f $fname Package)"
	echo "Extracting $dir..."
    	dpkg -x "$fname" "$dir"
}

function download_deb(){
	package="$1"
	echo "Downloading $package"
	apt download "$package"
}

for f in $(cat "$PACKAGES_SRC")
do
	echo "Testing package $f"
	if [[ $(check_package "$f") > 0 ]]; then
		echo "Package exists..."
		continue
	fi
	echo "Downloading package $f and -$PACKAGE_TYPE version"
	download_deb "$f"
	download_deb "${f:0:-$TYPE_LEN}"

	for pkg_file in $(find . -maxdepth 1 -iname "${f:0:-$TYPE_LEN}*.deb")
	do
		echo "PKG FILE: $pkg_file"
		extract_deb "$pkg_file"
	done
done
