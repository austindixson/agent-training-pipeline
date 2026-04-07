#!/bin/bash
# Monitor MLX LM training progress

LOG_FILE="${1:-training_qwen35_4k.log}"

echo "Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo "="$(tput cols | tr -d '\n') | tr ' ' '='

while true; do
    clear
    echo "Training Monitor - $(date '+%H:%M:%S')"
    echo "="$(tput cols | tr -d '\n') | tr ' ' '='
    echo ""
    echo "Last 30 lines of $LOG_FILE:"
    echo "----------------------------------------"
    tail -30 "$LOG_FILE" 2>/dev/null || echo "Log file not found yet..."
    echo ""
    echo "Process status:"
    ps aux | grep mlx_lm.lora | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%, Time: %s\n", $2, $3, $4, $10}' || echo "  Not running"
    echo ""
    sleep 5
done
