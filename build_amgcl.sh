#!/bin/sh
# Builds the amgcl wrapper into build/amgcl.a with g++.
#   ./build_amgcl.sh            serial
#   ./build_amgcl.sh --openmp   threaded, build Odin code with -define:AMGCL_OPENMP=true to link the runtime (libgomp)
set -e

FLAGS=""
case "$1" in
	"") ;;
	--openmp) FLAGS="-fopenmp" ;;
	*) echo "usage: $0 [--openmp]"; exit 1 ;;
esac

mkdir -p build
g++ -std=c++17 -O2 $FLAGS -I./vendor -c c_wrappers/amgcl.cpp -o ./build/amgcl_wrapper.o
ar rcs ./build/amgcl.a ./build/amgcl_wrapper.o
