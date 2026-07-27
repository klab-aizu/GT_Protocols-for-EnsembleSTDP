#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc

import file_transfer_pb2
import file_transfer_pb2_grpc


class ReceiveState:
    def __init__(self, expected_files: int, logdir: Path) -> None:
        self.expected_files = expected_files
        self.logdir = logdir
        self.lock = threading.Lock()
        self.received_files = 0
        self.received_paths = []
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def mark_success(self, saved_path: str) -> None:
        with self.lock:
            self.received_files += 1
            self.received_paths.append(saved_path)
            if self.received_files >= self.expected_files and self.finished_at is None:
                self.finished_at = time.time()
                self.write_summary()

    def write_summary(self) -> None:
        payload = {
            "expected_files": self.expected_files,
            "received_files": self.received_files,
            "started_at_unix": self.started_at,
            "finished_at_unix": self.finished_at,
            "batch_transfer_latency_sec": None if self.finished_at is None else self.finished_at - self.started_at,
            "received_paths": self.received_paths,
        }
        out = self.logdir / "grpc_receive_summary.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FileTransferService(file_transfer_pb2_grpc.FileTransferServiceServicer):
    def __init__(self, outdir: Path, state: ReceiveState, reject_duplicates: bool) -> None:
        self.outdir = outdir
        self.state = state
        self.reject_duplicates = reject_duplicates
        self.file_lock = threading.Lock()

    def UploadFile(self, request_iterator, context):
        info = None
        outfile = None
        tmp_path = None
        final_path = None
        hasher = hashlib.sha256()
        bytes_received = 0

        try:
            for req in request_iterator:
                which = req.WhichOneof("payload")

                if which == "info":
                    if info is not None:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("FileInfo sent more than once")
                        return file_transfer_pb2.UploadStatus(ok=False, message="duplicate FileInfo")

                    info = req.info
                    final_path = self.outdir / info.filename
                    tmp_path = self.outdir / f".{info.filename}.part"

                    if self.reject_duplicates and final_path.exists():
                        context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                        context.set_details(f"File already exists: {info.filename}")
                        return file_transfer_pb2.UploadStatus(
                            ok=False,
                            message="duplicate file",
                            saved_path=str(final_path),
                        )

                    with self.file_lock:
                        tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    outfile = open(tmp_path, "wb")

                elif which == "chunk_data":
                    if info is None or outfile is None:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("chunk_data received before FileInfo")
                        return file_transfer_pb2.UploadStatus(ok=False, message="chunk before info")

                    chunk = req.chunk_data
                    outfile.write(chunk)
                    hasher.update(chunk)
                    bytes_received += len(chunk)

            if info is None or outfile is None or tmp_path is None or final_path is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No file content received")
                return file_transfer_pb2.UploadStatus(ok=False, message="no file data")

            outfile.flush()
            outfile.close()
            outfile = None

            digest = hasher.hexdigest()
            if info.sha256 and digest != info.sha256:
                tmp_path.unlink(missing_ok=True)
                context.set_code(grpc.StatusCode.DATA_LOSS)
                context.set_details("SHA256 mismatch")
                return file_transfer_pb2.UploadStatus(ok=False, message="sha256 mismatch", sha256=digest)

            os.replace(tmp_path, final_path)
            self.state.mark_success(str(final_path))

            logging.info(
                "saved sender=%s file=%s bytes=%d sha256=%s",
                info.sender_name,
                info.filename,
                bytes_received,
                digest,
            )

            return file_transfer_pb2.UploadStatus(
                ok=True,
                message="saved",
                saved_path=str(final_path),
                bytes_received=bytes_received,
                sha256=digest,
            )

        except Exception as exc:
            logging.exception("upload failed")
            if outfile is not None:
                outfile.close()
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return file_transfer_pb2.UploadStatus(ok=False, message=str(exc))


def parse_args():
    p = argparse.ArgumentParser(description="gRPC batch receiver for .pth files")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--outdir", default="grpc_received")
    p.add_argument("--logdir", default="grpc_logs")
    p.add_argument("--expected-files", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--max-message-length", type=int, default=128 * 1024 * 1024)
    p.add_argument("--reject-duplicates", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(logdir / "grpc_receiver.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    state = ReceiveState(expected_files=args.expected_files, logdir=logdir)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_send_message_length", args.max_message_length),
            ("grpc.max_receive_message_length", args.max_message_length),
        ],
    )

    file_transfer_pb2_grpc.add_FileTransferServiceServicer_to_server(
        FileTransferService(outdir=outdir, state=state, reject_duplicates=args.reject_duplicates),
        server,
    )

    bind_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(bind_addr)
    server.start()

    logging.info(
        "gRPC receiver started on %s expected_files=%d outdir=%s",
        bind_addr,
        args.expected_files,
        outdir,
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("receiver stopped by user")
        server.stop(grace=1)


if __name__ == "__main__":
    main()
