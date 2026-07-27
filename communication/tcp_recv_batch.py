#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path

CHUNK = 1024 * 1024  # 1MB

def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data += chunk
    return data

def recv_json(conn):
    raw_len = recv_exact(conn, 8)
    (msg_len,) = struct.unpack("!Q", raw_len)
    payload = recv_exact(conn, msg_len)
    return json.loads(payload.decode("utf-8"))

def send_json(conn, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    conn.sendall(struct.pack("!Q", len(payload)))
    conn.sendall(payload)

def sanitize_name(name):
    return os.path.basename(name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
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

    per_file_log = logdir / "tcp_receive_file_log.jsonl"
    summary_log = logdir / "tcp_receive_summary.json"

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(32)

    print(f"[RECEIVER] listening on {args.host}:{args.port}")
    print(f"[RECEIVER] expecting {args.expected_files} files")

    received_count = 0
    batch_started_at = None
    batch_finished_at = None
    received_names = set()

    while received_count < args.expected_files:
        conn, addr = server.accept()
        with conn:
            file_started_at = time.time()
            if batch_started_at is None:
                batch_started_at = file_started_at

            meta = recv_json(conn)
            sender = meta["sender"]
            filename = sanitize_name(meta["filename"])
            filesize = int(meta["filesize"])
            sent_sha256 = meta["sha256"]

            save_dir = outdir / sender
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename

            if args.reject_duplicates and filename in received_names:
                send_json(conn, {
                    "ok": False,
                    "error": f"duplicate filename rejected: {filename}"
                })
                print(f"[REJECT] duplicate filename: {filename}")
                continue

            sha = hashlib.sha256()
            bytes_received = 0

            with open(save_path, "wb") as f:
                while bytes_received < filesize:
                    want = min(CHUNK, filesize - bytes_received)
                    chunk = conn.recv(want)
                    if not chunk:
                        raise ConnectionError(
                            f"Connection lost while receiving {filename} "
                            f"({bytes_received}/{filesize} bytes)"
                        )
                    f.write(chunk)
                    sha.update(chunk)
                    bytes_received += len(chunk)
                f.flush()
                os.fsync(f.fileno())

            recv_sha256 = sha.hexdigest()
            ok = (recv_sha256 == sent_sha256 and bytes_received == filesize)

            file_finished_at = time.time()
            record = {
                "sender": sender,
                "filename": filename,
                "filesize": filesize,
                "bytes_received": bytes_received,
                "sender_sha256": sent_sha256,
                "receiver_sha256": recv_sha256,
                "sha256_match": recv_sha256 == sent_sha256,
                "ok": ok,
                "client_addr": addr[0],
                "client_port": addr[1],
                "started_at_unix": file_started_at,
                "finished_at_unix": file_finished_at,
                "transfer_latency_sec": round(file_finished_at - file_started_at, 6),
                "saved_to": str(save_path),
            }

            with open(per_file_log, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(record, ensure_ascii=False) + "\n")

            send_json(conn, {
                "ok": ok,
                "filename": filename,
                "receiver_sha256": recv_sha256,
                "transfer_latency_sec": round(file_finished_at - file_started_at, 6),
            })

            if ok:
                received_count += 1
                received_names.add(filename)
                print(f"[OK] {received_count}/{args.expected_files} {sender}/{filename} "
                      f"{bytes_received} bytes in {file_finished_at - file_started_at:.3f}s")
            else:
                print(f"[FAIL] hash mismatch for {sender}/{filename}")

    batch_finished_at = time.time()
    summary = {
        "expected_files": args.expected_files,
        "received_files": received_count,
        "batch_started_at_unix": batch_started_at,
        "batch_finished_at_unix": batch_finished_at,
        "batch_transfer_latency_sec": round(batch_finished_at - batch_started_at, 6),
        "outdir": str(outdir),
        "per_file_log": str(per_file_log),
    }

    with open(summary_log, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print("[RECEIVER] completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    server.close()

if __name__ == "__main__":
    main()
