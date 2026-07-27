#!/bin/bash
set -euo pipefail

PROTOCOL="${1:?Usage: $0 tcp|zmq|grpc TRIAL_NUMBER}"
TRIAL="${2:?Usage: $0 tcp|zmq|grpc TRIAL_NUMBER}"

BASE="$HOME/snn/EnsembleSTDP/scripts/mnist"
RECEIVER_IP="192.168.100.100"

NODES=(
  "vm-node2"
  "vm-node3"
  "vm-node4"
  "vm-node5"
  "vm-node6"
  "vm-node7"
  "vm-node8"
)

IPS=(
  "192.168.100.141"
  "192.168.100.143"
  "192.168.100.152"
  "192.168.100.151"
  "192.168.100.154"
  "192.168.100.153"
  "192.168.100.129"
)

MODELS=(1 2 3 4 5 6 7)
STARTS=(0 6000 12000 18000 24000 30000 36000)
ENDS=(6000 12000 18000 24000 30000 36000 42000)

cd "$BASE"
source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate

case "$PROTOCOL" in

tcp)
    OUTDIR="tcp_received_sequential_trial${TRIAL}"
    LOGDIR="tcp_logs_sequential_trial${TRIAL}"

    rm -rf "$OUTDIR" "$LOGDIR"
    mkdir -p "$OUTDIR" "$LOGDIR"

    python3 tcp_recv_batch.py \
      --host 0.0.0.0 \
      --port 5555 \
      --expected-files 21 \
      --outdir "$OUTDIR" \
      --logdir "$LOGDIR" &

    RECEIVER_PID=$!
    sleep 2

    for i in "${!NODES[@]}"; do
        NODE="${NODES[$i]}"
        IP="${IPS[$i]}"
        MODEL="${MODELS[$i]}"
        START="${STARTS[$i]}"
        END="${ENDS[$i]}"

        echo "===== TCP: starting $NODE ====="

        ssh "hayato@$IP" "
          cd ~/snn/EnsembleSTDP/scripts/mnist &&
          source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate;
          rm -rf send_tcp;
          mkdir -p send_tcp tcp_logs_sequential_trial${TRIAL};
          cp models/model_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;
          cp models/assignments_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;
          cp models/proportions_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;
          python3 tcp_send_all.py \
            --receiver-ip $RECEIVER_IP \
            --receiver-port 5555 \
            --sender-name $NODE \
            --folder send_tcp \
            --sleep-between 0 \
            --logfile tcp_logs_sequential_trial${TRIAL}/${NODE}_send.jsonl
        "

        echo "===== TCP: finished $NODE ====="
    done

    wait "$RECEIVER_PID"
    ;;

zmq)
    OUTDIR="zmq_received_sequential_trial${TRIAL}"
    LOGDIR="zmq_logs_sequential_trial${TRIAL}"

    rm -rf "$OUTDIR" "$LOGDIR"
    mkdir -p "$OUTDIR" "$LOGDIR"

    python3 zmq_recv_batch.py \
      --bind-ip 0.0.0.0 \
      --port 5555 \
      --expected-files 21 \
      --outdir "$OUTDIR" \
      --logdir "$LOGDIR" &

    RECEIVER_PID=$!
    sleep 2

    for i in "${!NODES[@]}"; do
        NODE="${NODES[$i]}"
        IP="${IPS[$i]}"
        MODEL="${MODELS[$i]}"
        START="${STARTS[$i]}"
        END="${ENDS[$i]}"

        echo "===== ZeroMQ: starting $NODE ====="

        ssh "hayato@$IP" "
          cd ~/snn/EnsembleSTDP/scripts/mnist &&
          source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate;
          rm -rf send_zmq;
          mkdir -p send_zmq;
          cp models/model_${MODEL}_100_neurons_${START}_to_${END}.pth send_zmq/;
          cp models/assignments_${MODEL}_100_neurons_${START}_to_${END}.pth send_zmq/;
          cp models/proportions_${MODEL}_100_neurons_${START}_to_${END}.pth send_zmq/;
          python3 zmq_send_batch.py \
            --receiver-ip $RECEIVER_IP \
            --port 5555 \
            --indir send_zmq
        "

        echo "===== ZeroMQ: finished $NODE ====="
    done

    wait "$RECEIVER_PID"
    ;;

grpc)
    # Kill any stale gRPC receiver from previous trials.
    pkill -9 -f grpc_recv_batch.py 2>/dev/null || true
    sleep 2

    OUTDIR="grpc_received_sequential_trial${TRIAL}"
    LOGDIR="grpc_logs_sequential_trial${TRIAL}"

    rm -rf "$OUTDIR" "$LOGDIR"
    mkdir -p "$OUTDIR" "$LOGDIR"

    python3 grpc_recv_batch.py \
      --host 0.0.0.0 \
      --port 50051 \
      --outdir "$OUTDIR" \
      --logdir "$LOGDIR" \
      --expected-files 21 &

    RECEIVER_PID=$!
    sleep 2

    for i in "${!NODES[@]}"; do
        NODE="${NODES[$i]}"
        IP="${IPS[$i]}"
        MODEL="${MODELS[$i]}"
        START="${STARTS[$i]}"
        END="${ENDS[$i]}"

        echo "===== gRPC: starting $NODE ====="

        ssh "hayato@$IP" "
          cd ~/snn/EnsembleSTDP/scripts/mnist &&
          source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate;
          rm -rf send_grpc;
          mkdir -p send_grpc grpc_logs_sequential_trial${TRIAL};
          cp models/model_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;
          cp models/assignments_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;
          cp models/proportions_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;
          python3 grpc_send_batch.py \
            --server-ip $RECEIVER_IP \
            --port 50051 \
            --input-dir send_grpc \
            --pattern '*.pth' \
            --sender-name $NODE \
            --log-file grpc_logs_sequential_trial${TRIAL}/${NODE}_send.jsonl
        "

        echo "===== gRPC: finished $NODE ====="
    done

    # gRPC receiver does not terminate automatically.
    # Wait until all 21 files are received, then stop it automatically.
    for i in {1..30}; do
        RECEIVED=$(find "$OUTDIR" -type f -name "*.pth" | wc -l)

        if [ "$RECEIVED" -eq 21 ]; then
            echo "All 21 files received."
            break
        fi

        sleep 1
    done

    RECEIVED=$(find "$OUTDIR" -type f -name "*.pth" | wc -l)

    if [ "$RECEIVED" -ne 21 ]; then
        echo "ERROR: Expected 21 files, but received $RECEIVED."
        kill -INT "$RECEIVER_PID" 2>/dev/null || true
        wait "$RECEIVER_PID" 2>/dev/null || true
        exit 1
    fi

    # Stop gRPC receiver automatically after all 21 files are received.
    kill -TERM "$RECEIVER_PID" 2>/dev/null || true
    sleep 2

    if kill -0 "$RECEIVER_PID" 2>/dev/null; then
        echo "gRPC receiver still running. Force stopping..."
        kill -KILL "$RECEIVER_PID" 2>/dev/null || true
    fi

    wait "$RECEIVER_PID" 2>/dev/null || true

    echo "gRPC receiver stopped automatically."
    ;;

*)
    echo "Unknown protocol: $PROTOCOL"
    exit 1
    ;;
esac

echo
echo "========================================"
echo "Protocol: $PROTOCOL"
echo "Trial: $TRIAL"
echo "Sequential transfer completed."
echo "========================================"
