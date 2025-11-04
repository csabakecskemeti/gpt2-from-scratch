#!/bin/bash
# Start TensorBoard to monitor training
# Usage: ./start_tensorboard.sh [port] [logdir]

PORT="${1:-6006}"
LOGDIR="${2:-./runs}"

echo "============================================================"
echo "Starting TensorBoard"
echo "============================================================"
echo "Port: $PORT"
echo "Log directory: $LOGDIR"
echo ""
echo "Access dashboard at: http://localhost:$PORT"
echo ""
echo "If running remotely, use SSH tunnel:"
echo "  ssh -L $PORT:localhost:$PORT user@hostname"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "============================================================"
echo ""

tensorboard --logdir="$LOGDIR" --bind_all --port="$PORT"

