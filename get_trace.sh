#!/bin/bash

hawkeye_path="./hawkeye/hawkeye_final.cc"
lru_path="./example/lru.cc"
random_path="./random/random.cc"

hawkeye_out="hawkeye_trace"
lru_out="lru_trace"
random_out="random_trace"

trace=$1
compute_belady=true

#Compile and run Hawkeye
echo "Compiling Hawkeye..."
g++ -Wall --std=c++11 -o hawkeye-config1 $hawkeye_path lib/config1.a

echo "Running Hawkeye..."
./hawkeye-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > "$hawkeye_out.txt"

echo "Format Hawkeye trace as CSV..."
python convert_to_csv.py -i "$hawkeye_out.txt" -o "$hawkeye_out.csv"
rm "$hawkeye_out.txt"

if [ "$compute_belady" = true ]; then
	echo "Include Belady Optimal Solution in trace..."
	python trace_with_belady.py -i "$hawkeye_out.csv" -o "$hawkeye_out""_belady.csv"
	rm "$hawkeye_out.csv"
fi

#Compile and run LRU
echo "Compiling LRU..."
g++ -Wall --std=c++11 -o lru-config1 $lru_path lib/config1.a

echo "Running LRU..."
./lru-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > "$lru_out.txt"

echo "Format LRU trace as CSV..."
python convert_to_csv.py -i "$lru_out.txt" -o "$lru_out.csv"
rm "$lru_out.txt"

if [ "$compute_belady" = true ]; then
	echo "Include Belady Optimal Solution in trace..."
	python trace_with_belady.py -i "$lru_out.csv" -o "$lru_out""_belady.csv"
	rm "$lru_out.csv"
fi

#Compile and run LRU
echo "Compiling Random..."
g++ -Wall --std=c++11 -o random-config1 $random_path lib/config1.a

echo "Running Random..."
./random-config1 -warmup_instructions 1000000 -simulation_instructions 10000000 -traces $trace > "$random_out.txt"

echo "Format Random trace as CSV..."
python convert_to_csv.py -i "$random_out.txt" -o "$random_out.csv"
rm "$random_out.txt"

if [ "$compute_belady" = true ]; then
	echo "Include Belady Optimal Solution in trace..."
	python trace_with_belady.py -i "$random_out.csv" -o "$random_out""_belady.csv"
	rm "$random_out.csv"
fi
