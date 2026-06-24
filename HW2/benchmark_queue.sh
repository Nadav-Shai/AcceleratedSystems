#!/bin/bash

THREADS=$1  # pass thread count as argument, e.g. ./benchmark_queue.sh 1024
OUTFILE="results_queue_${THREADS}.txt"

echo "=== Queue Server Benchmark (threads=$THREADS) ===" | tee -a $OUTFILE
echo "" | tee -a $OUTFILE

# Step 1: measure maxLoad at load=0
echo "--- load=0 (maxLoad measurement) ---" | tee -a $OUTFILE
OUTPUT=$(./ex2 queue $THREADS 0 2>/dev/null)
echo "$OUTPUT" | tee -a $OUTFILE
MAX_LOAD=$(echo "$OUTPUT" | grep "throughput" | awk '{print $3}' | cut -d'.' -f1)
echo "" | tee -a $OUTFILE
echo "Detected maxLoad = $MAX_LOAD req/sec" | tee -a $OUTFILE
echo "" | tee -a $OUTFILE

# Step 2: vary load from maxLoad/10 to 2*maxLoad in 10 steps
echo "--- Sweep ---" | tee -a $OUTFILE
for i in $(seq 1 10); do
    LOAD=$(echo "$MAX_LOAD * $i / 10" | bc)
    echo "--- load=$LOAD ---" | tee -a $OUTFILE
    ./ex2 queue $THREADS $LOAD 2>/dev/null | tee -a $OUTFILE
    echo "" | tee -a $OUTFILE
done

# Step 3: 2*maxLoad
LOAD=$(echo "$MAX_LOAD * 2" | bc)
echo "--- load=$LOAD (2x maxLoad) ---" | tee -a $OUTFILE
./ex2 queue $THREADS $LOAD 2>/dev/null | tee -a $OUTFILE
echo "" | tee -a $OUTFILE

echo "=== Done. Results saved to $OUTFILE ==="