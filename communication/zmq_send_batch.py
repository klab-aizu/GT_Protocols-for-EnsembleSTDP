#!/usr/bin/env python3
import argparse
import json
import socket
import time
from pathlib import Path

import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-ip", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--indir", required=True)
    args = parser.parse_args()

    indir = Path(args.indir)
    files = sorted([p for p in indir.iterdir() if p.is_file() and p.suffix == ".pth"])

    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect(f"tcp://{args.receiver_ip}:{args.port}")

    sender_name = socket.gethostname()
    results = []

    try:
        for path in files:
            with open(path, "rb") as f:
                data = f.read()

            sent_ts = time.time()
            payload = {
                "filename": path.name,
                "filehex": data.hex(),
                "sender": sender_name,
                "sent_ts": sent_ts,
            }

            sock.send_json(payload)
            reply = sock.recv_json()

            results.append(
                {
                    "filename": path.name,
                    "bytes": len(data),
                    "sent_ts": sent_ts,
                    "reply": reply,
                }
            )

    finally:
        sock.close(0)
        context.term()

    print(json.dumps({"sent_files": results}, indent=2))


if __name__ == "__main__":
    main()
