#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

import zmq


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--reject-duplicates", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.bind_ip}:{args.port}")

    received_count = 0
    received_files = []
    seen_hashes = set()

    batch_start = None
    batch_end = None

    try:
        while received_count < args.expected_files:
            msg = socket.recv_json()

            filename = msg["filename"]
            filehex = msg["filehex"]
            sender = msg.get("sender", "unknown")
            sent_ts = msg.get("sent_ts", None)

            data = bytes.fromhex(filehex)
            digest = sha256_bytes(data)

            if batch_start is None:
                batch_start = time.time()

            save_path = outdir / filename
            is_duplicate = digest in seen_hashes

            if args.reject_duplicates and is_duplicate:
                socket.send_json(
                    {
                        "status": "duplicate_rejected",
                        "filename": filename,
                        "sha256": digest,
                    }
                )
                continue

            with open(save_path, "wb") as f:
                f.write(data)

            seen_hashes.add(digest)
            recv_ts = time.time()
            received_count += 1

            record = {
                "index": received_count,
                "filename": filename,
                "bytes": len(data),
                "sha256": digest,
                "sender": sender,
                "sent_ts": sent_ts,
                "recv_ts": recv_ts,
                "save_path": str(save_path),
            }
            received_files.append(record)

            socket.send_json(
                {
                    "status": "ok",
                    "filename": filename,
                    "sha256": digest,
                    "received_count": received_count,
                }
            )

        batch_end = time.time()

    finally:
        socket.close(0)
        context.term()

    summary = {
        "protocol": "ZeroMQ",
        "expected_files": args.expected_files,
        "received_files_count": received_count,
        "batch_start_ts": batch_start,
        "batch_end_ts": batch_end,
        "batch_transfer_latency_sec": None
        if batch_start is None or batch_end is None
        else batch_end - batch_start,
        "files": received_files,
    }

    summary_path = logdir / "zmq_receive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
