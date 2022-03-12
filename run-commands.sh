#!/bin/sh

# # Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake ..
make 
cd .. 
# cuda-memcheck ./build/wc
./build/slab
./build/wc
