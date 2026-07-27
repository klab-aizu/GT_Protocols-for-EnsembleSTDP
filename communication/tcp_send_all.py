#!/usr/bin/env python3
import argparse
import hashlib
import json
import socket
import struct
import time
from pathlib import Path

CHUNK = 1024 * 1024  # 1MB

def send_json(sock, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving ACK")
        data += chunk
    return data

def recv_json(sock):
    raw_len = recv_exact(sock, 8)
    (msg_len,) = struct.unpack("!Q", raw_len)
    payload = recv_exact(sock, msg_len)
    return json.loads(payload.decode("utf-8"))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def send_one_file(receiver_ip, receiver_port, sender_name, path, timeout_sec):
    filesize = path.stat().st_size
    digest = sha256_file(path)

    meta = {
        "sender": sender_name,
        "filename": path.name,
        "filesize": filesize,
        "sha256": digest,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)

    started_at = time.time()
    sock.connect((receiver_ip, receiver_port))
    send_json(sock, meta)

    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            sock.sendall(chunk)

    ack = recv_json(sock)
    finished_at = time.time()
    sock.close()

    return {
        "sender": sender_name,
        "filename": path.name,
        "filesize": filesize,
        "sha256": digest,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "sender_transfer_latency_sec": round(finished_at - started_at, 6),
        "ack": ack,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-ip", required=True)
    parser.add_argument("--receiver-port", type=int, default=5555)
    parser.add_argument("--sender-name", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--sleep-between", type=float, default=0.3)
    parser.add_argument("--logfile", default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.pth"))
    if not files:
        raise SystemExit(f"No .pth files found in {folder}")

    logfile = Path(args.logfile) if args.logfile else folder / f"tcp_send_log_{args.sender_name}.jsonl"

    print(f"[SENDER] sender={args.sender_name}")
    print(f"[SENDER] folder={folder}")
    print(f"[SENDER] files={len(files)}")

    for path in files:
        result = send_one_file(
            receiver_ip=args.receiver_ip,
            receiver_port=args.receiver_port,
            sender_name=args.sender_name,
            path=path,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps(result, ensure_ascii=False))
        with open(logfile, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(result, ensure_ascii=False) + "\n")
        time.sleep(args.sleep_between)

    print(f"[SENDER] done -> {logfile}")

if __name__ == "__main__":
    main()
