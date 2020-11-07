#!/bin/bash

hawkeye_path="./hawkeye/hawkeye_final.cc"
lru_path="./example/lru.cc"

hawkeye_out="hawkeye_trace"
lru_out="lru_trace"

trace="trace/bzip2_10M.trace.gz"

#Compile and run Hawkeye
g++ -Wall --std=c++11 -o hawkeye-config1 $hawkeye_path lib/config1.a
./hawkeye-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > "$hawkeye_out.txt"

python convert_to_csv.py -i "$hawkeye_out.txt" -o "$hawkeye_out.csv"
rm "$hawkeye_out.txt"

#Compile and run LRU
g++ -Wall --std=c++11 -o lru-config1 $lru_path lib/config1.a
./hawkeye-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > "$lru_out.txt"

python convert_to_csv.py -i "$lru_out.txt" -o "$lru_out.csv"
rm "$lru_out.txt"



