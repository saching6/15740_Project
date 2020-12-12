#!/bin/bash

LDFLAGS="-Wl,--copy-dt-needed-entries"
LD_LIBRARY_PATH="/home/josh/Desktop/libtorch/lib/:$LD_LIBRARY_PATH"
LIB_PATH="/home/josh/Desktop/libtorch/lib/"
echo $LD_LIBRARY_PATH

trace=$1

echo "Compiling Deep Learning Cache Replacement..."

g++ -Wall -std=c++14 -o belady_dl ./dl_cache/belady_dl.cc -I /home/josh/Desktop/libtorch/include/torch/csrc/api/include/ -I /home/josh/Desktop/libtorch/include/  -L /home/josh/Desktop/libtorch/lib/ -lc10 -ltorch_cpu -D_GLIBCXX_USE_CXX11_ABI=0 -Wl,-rpath "$LIB_PATH" lib/config1.a

echo "Running Deep Learning Cache Replacement..."
./belady_dl -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace

