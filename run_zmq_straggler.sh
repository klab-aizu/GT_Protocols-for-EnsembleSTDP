#!/usr/bin/env bash

DELAY="$1"
TRIAL_TAG="$2"
RECEIVER_IP="192.168.100.100"

run_sender() {
    IP="$1"
    IDX="$2"
    START="$3"
    END="$4"
    EXTRA_DELAY="$5"

    ssh -o BatchMode=yes hayato@"$IP" bash -s -- \
        "$IDX" "$START" "$END" "$RECEIVER_IP" "$EXTRA_DELAY" "$TRIAL_TAG" <<'REMOTE' &

IDX="$1"
START="$2"
END="$3"
RECEIVER_IP="$4"
EXTRA_DELAY="$5"
TRIAL_TAG="$6"

cd ~/snn/EnsembleSTDP/scripts/mnist || exit 1
source ../../snn-env/bin/activate 2>/dev/null || source ~/snn-env/bin/activate

SEND_DIR="send_zmq_straggler_${TRIAL_TAG}"

rm -rf "$SEND_DIR"
mkdir -p "$SEND_DIR"

cp "models/model_${IDX}_100_neurons_${START}_to_${END}.pth" "$SEND_DIR"/
cp "models/assignments_${IDX}_100_neurons_${START}_to_${END}.pth" "$SEND_DIR"/
cp "models/proportions_${IDX}_100_neurons_${START}_to_${END}.pth" "$SEND_DIR"/

sleep "$EXTRA_DELAY"

python3 zmq_send_batch.py \
  --receiver-ip "$RECEIVER_IP" \
  --port 5555 \
  --indir "$SEND_DIR"
REMOTE
}

run_sender 192.168.100.141 1 0 6000 0
run_sender 192.168.100.143 2 6000 12000 0
run_sender 192.168.100.152 3 12000 18000 "$DELAY"
run_sender 192.168.100.151 4 18000 24000 0
run_sender 192.168.100.154 5 24000 30000 0
run_sender 192.168.100.153 6 30000 36000 0
run_sender 192.168.100.129 7 36000 42000 0

wait
echo "ZeroMQ Straggler ${TRIAL_TAG}, vm-node4 delay=${DELAY}s finished"
