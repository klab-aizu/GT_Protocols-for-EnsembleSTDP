#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import grpc

import file_transfer_pb2
import file_transfer_pb2_grpc


def sha256_of_file(path: Path, chunk_size: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_file_chunks(path: Path, sender_name: str, chunk_size: int):
    stat = path.stat()
    digest = sha256_of_file(path, chunk_size)

    yield file_transfer_pb2.FileChunk(
        info=file_transfer_pb2.FileInfo(
            filename=path.name,
            filesize=stat.st_size,
            sha256=digest,
            sender_name=sender_name,
        )
    )

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield file_transfer_pb2.FileChunk(chunk_data=chunk)


def upload_one(stub, path: Path, sender_name: str, timeout_sec: int, chunk_size: int):
    started = time.time()
    response = stub.UploadFile(iter_file_chunks(path, sender_name, chunk_size), timeout=timeout_sec)
    elapsed = time.time() - started

    return {
        "filename": path.name,
        "ok": response.ok,
        "message": response.message,
        "saved_path": response.saved_path,
        "bytes_received": response.bytes_received,
        "sha256": response.sha256,
        "elapsed_sec": elapsed,
    }


def parse_args():
    p = argparse.ArgumentParser(description="gRPC batch sender for .pth files")
    p.add_argument("--server-ip", required=True)
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--input-dir", default="send_grpc")
    p.add_argument("--pattern", default="*.pth")
    p.add_argument("--sender-name", default=socket.gethostname())
    p.add_argument("--chunk-size", type=int, default=1024 * 1024)
    p.add_argument("--timeout-sec", type=int, default=600)
    p.add_argument("--log-file", default="grpc_send_results.json")
    p.add_argument("--max-message-length", type=int, default=128 * 1024 * 1024)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {input_dir}")

    target = f"{args.server_ip}:{args.port}"

    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", args.max_message_length),
            ("grpc.max_receive_message_length", args.max_message_length),
        ],
    )
    stub = file_transfer_pb2_grpc.FileTransferServiceStub(channel)

    batch_started = time.time()
    results = []

    for path in files:
        result = upload_one(stub, path, args.sender_name, args.timeout_sec, args.chunk_size)
        print(f"{path.name}: ok={result['ok']} elapsed={result['elapsed_sec']:.3f}s message={result['message']}")
        results.append(result)

    batch_elapsed = time.time() - batch_started

    payload = {
        "server": target,
        "sender_name": args.sender_name,
        "input_dir": str(input_dir),
        "files_sent": len(files),
        "batch_started_at_unix": batch_started,
        "batch_elapsed_sec": batch_elapsed,
        "results": results,
    }

    Path(args.log_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"batch_elapsed_sec={batch_elapsed:.3f}")
    print(f"log_file={args.log_file}")


if __name__ == "__main__":
    main()
