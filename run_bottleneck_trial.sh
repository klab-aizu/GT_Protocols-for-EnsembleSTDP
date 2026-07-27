#!/bin/bash
set -euo pipefail

PROTOCOL="${1:?Usage: $0 tcp|zmq|grpc TRIAL_NUMBER [full|transfer-only]}"
TRIAL="${2:?Usage: $0 tcp|zmq|grpc TRIAL_NUMBER [full|transfer-only]}"
RUN_MODE="${3:-full}"

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
  "vm-node9"
  "vm-node10"
)

IPS=(
  "192.168.100.141"
  "192.168.100.143"
  "192.168.100.152"
  "192.168.100.151"
  "192.168.100.154"
  "192.168.100.153"
  "192.168.100.129"
  "192.168.100.157"
  "192.168.100.44"
)

MODELS=(1 2 3 4 5 6 7 8 9)

STARTS=(
  0
  6000
  12000
  18000
  24000
  30000
  36000
  42000
  48000
)

ENDS=(
  6000
  12000
  18000
  24000
  30000
  36000
  42000
  48000
  54000
)

cd "$BASE"
source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate


############################################################
# TCP
############################################################

if [ "$PROTOCOL" = "tcp" ]; then

  OUTDIR="tcp_received_bottleneck_trial${TRIAL}"
  LOGDIR="tcp_logs_bottleneck_trial${TRIAL}"

  rm -rf "$OUTDIR" "$LOGDIR"
  mkdir -p "$OUTDIR" "$LOGDIR"

  python3 tcp_recv_batch.py \
    --host 0.0.0.0 \
    --port 5555 \
    --expected-files 27 \
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

    echo "===== TCP: $NODE / model_$MODEL ====="

    ssh "hayato@$IP" "
      cd ~/snn/EnsembleSTDP/scripts/mnist &&
      source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate;

      rm -rf send_tcp;
      mkdir -p send_tcp tcp_logs_bottleneck_trial${TRIAL};

      cp models/model_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;
      cp models/assignments_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;
      cp models/proportions_${MODEL}_100_neurons_${START}_to_${END}.pth send_tcp/;

      python3 tcp_send_all.py \
        --receiver-ip $RECEIVER_IP \
        --receiver-port 5555 \
        --sender-name $NODE \
        --folder send_tcp \
        --sleep-between 0 \
        --logfile tcp_logs_bottleneck_trial${TRIAL}/${NODE}_send.jsonl
    "

  done

  wait "$RECEIVER_PID"


############################################################
# ZeroMQ
############################################################

elif [ "$PROTOCOL" = "zmq" ]; then

  OUTDIR="zmq_received_bottleneck_trial${TRIAL}"
  LOGDIR="zmq_logs_bottleneck_trial${TRIAL}"

  rm -rf "$OUTDIR" "$LOGDIR"
  mkdir -p "$OUTDIR" "$LOGDIR"

  python3 zmq_recv_batch.py \
    --bind-ip 0.0.0.0 \
    --port 5555 \
    --expected-files 27 \
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

    echo "===== ZeroMQ: $NODE / model_$MODEL ====="

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

  done

  wait "$RECEIVER_PID"


############################################################
# gRPC
############################################################

elif [ "$PROTOCOL" = "grpc" ]; then

  pkill -9 -f grpc_recv_batch.py 2>/dev/null || true
  sleep 2

  OUTDIR="grpc_received_bottleneck_trial${TRIAL}"
  LOGDIR="grpc_logs_bottleneck_trial${TRIAL}"

  rm -rf "$OUTDIR" "$LOGDIR"
  mkdir -p "$OUTDIR" "$LOGDIR"

  python3 grpc_recv_batch.py \
    --host 0.0.0.0 \
    --port 50051 \
    --outdir "$OUTDIR" \
    --logdir "$LOGDIR" \
    --expected-files 27 &

  RECEIVER_PID=$!
  sleep 2

  for i in "${!NODES[@]}"; do

    NODE="${NODES[$i]}"
    IP="${IPS[$i]}"
    MODEL="${MODELS[$i]}"
    START="${STARTS[$i]}"
    END="${ENDS[$i]}"

    echo "===== gRPC: $NODE / model_$MODEL ====="

    ssh "hayato@$IP" "
      cd ~/snn/EnsembleSTDP/scripts/mnist &&
      source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate;

      rm -rf send_grpc;
      mkdir -p send_grpc grpc_logs_bottleneck_trial${TRIAL};

      cp models/model_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;
      cp models/assignments_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;
      cp models/proportions_${MODEL}_100_neurons_${START}_to_${END}.pth send_grpc/;

      python3 grpc_send_batch.py \
        --server-ip $RECEIVER_IP \
        --port 50051 \
        --input-dir send_grpc \
        --pattern '*.pth' \
        --sender-name $NODE \
        --log-file grpc_logs_bottleneck_trial${TRIAL}/${NODE}_send.jsonl
    "

  done

  for i in {1..30}; do

    RECEIVED=$(find "$OUTDIR" -type f -name "*.pth" | wc -l)

    if [ "$RECEIVED" -eq 27 ]; then
      echo "All 27 files received."
      break
    fi

    sleep 1
  done

  RECEIVED=$(find "$OUTDIR" -type f -name "*.pth" | wc -l)

  if [ "$RECEIVED" -ne 27 ]; then
    echo "ERROR: Expected 27 files, received $RECEIVED."
    kill -TERM "$RECEIVER_PID" 2>/dev/null || true
    exit 1
  fi

  kill -TERM "$RECEIVER_PID" 2>/dev/null || true
  sleep 2

  if kill -0 "$RECEIVER_PID" 2>/dev/null; then
    kill -KILL "$RECEIVER_PID" 2>/dev/null || true
  fi

  wait "$RECEIVER_PID" 2>/dev/null || true

  echo "gRPC receiver stopped automatically."

else

  echo "Unknown protocol: $PROTOCOL"
  exit 1

fi


############################################################
# Validate 27 received files
############################################################

COUNT=$(find "$OUTDIR" -type f -name "*.pth" | wc -l)

echo
echo "========================================"
echo "Protocol: $PROTOCOL"
echo "Trial: $TRIAL"
echo "Received files: $COUNT"
echo "========================================"

if [ "$COUNT" -ne 27 ]; then
  echo "ERROR: Trial is invalid."
  exit 1
fi


############################################################
# Copy received files to models_to_merge
############################################################

rm -rf models_to_merge
mkdir -p models_to_merge

find "$OUTDIR" -type f -name "*.pth" -exec cp {} models_to_merge/ \;

MERGE_COUNT=$(find models_to_merge -type f -name "*.pth" | wc -l)

echo "models_to_merge files: $MERGE_COUNT"

if [ "$MERGE_COUNT" -ne 27 ]; then
  echo "ERROR: models_to_merge must contain 27 files."
  exit 1
fi


############################################################
# Transfer-only mode
############################################################

if [ "$RUN_MODE" = "transfer-only" ]; then
  echo
  echo "========================================"
  echo "TRANSFER-ONLY TRIAL COMPLETED"
  echo "Protocol: $PROTOCOL"
  echo "Trial: $TRIAL"
  echo "Received files: $COUNT"
  echo "========================================"
  exit 0
fi


############################################################
# Merge + Compression + Evaluation
############################################################

MERGE_LOG="${LOGDIR}/merge_output.txt"

echo
echo "Starting merge/compression/evaluation..."
echo

/usr/bin/time -f "TOTAL_WALL_TIME_SEC=%e" \
python3 eth_mnist_merge.py \
  --method mse \
  --n_compression 100 \
  --n_test 10000 \
2>&1 | tee "$MERGE_LOG"

echo
echo "========================================"
echo "BOTTLENECK TRIAL COMPLETED"
echo "Protocol: $PROTOCOL"
echo "Trial: $TRIAL"
echo "Transfer files: 27"
echo "Merge log: $MERGE_LOG"
echo "========================================"

