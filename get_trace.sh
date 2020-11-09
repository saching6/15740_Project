#!/bin/bash

hawkeye_path="./hawkeye/hawkeye_final.cc"
lru_path="./example/lru.cc"
random_path="./random/random.cc"


trace=$1
tracetag=$2

hawkeye_out="traces/hawkeye_trace_$tracetag.txt"
lru_out="traces/lru_trace_$tracetag.txt"
random_out="traces/random_trace_$tracetag.txt"


echo $hawkeye_out
echo $lru_out
echo $random_out

#Compile and run Hawkeye
echo "Compiling Hawkeye..."
g++ -Wall --std=c++11 -o hawkeye-config1 $hawkeye_path lib/config1.a

echo "Running Hawkeye..."
./hawkeye-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > $hawkeye_out

echo "Format Hawkeye trace as CSV..."
python convert_to_csv.py -i "$hawkeye_out" -o "$hawkeye_out.csv"
#rm "$hawkeye_out.txt"

#Compile and run LRU
echo "Compiling LRU..."
g++ -Wall --std=c++11 -o lru-config1 $lru_path lib/config1.a

echo "Running LRU..."
./lru-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > $lru_out

echo "Format LRU trace as CSV..."
python convert_to_csv.py -i "$lru_out" -o "$lru_out.csv"
#rm "$lru_out.txt"

#Compile and run LRU
echo "Compiling Random..."
g++ -Wall --std=c++11 -o random-config1 $random_path lib/config1.a

echo "Running Random..."
./random-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > $random_out

echo "Format Random trace as CSV..."
python convert_to_csv.py -i "$random_out" -o "$random_out.csv"
# rm "$random_out"


