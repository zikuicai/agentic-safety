#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"command to run\""
    exit 1
fi

# Create random string and timestamp
random_string=$(openssl rand -hex 4)
datetime_str=$(date +"%Y%m%d-%H%M%S")
current_time=$(date +%s)

# Setup logs directory
logs_dir="$(pwd)/logs/general"
mkdir -p "$logs_dir"

# Generate log filename
log_file="$logs_dir/${current_time}-${random_string}-${datetime_str}.log"

echo "Log file: $log_file"

# Execute the command and redirect output
eval "$@" > "$log_file" 2>&1 &

# Print the PID of the background process
pid=$!
echo "PID: $pid"
