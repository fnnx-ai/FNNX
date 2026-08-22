# client.py
import atexit
import itertools
import json
import queue
import subprocess
import threading
from collections import deque


class StdIOClient:
    def __init__(self, server_cmd: list[str]):
        self.proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._write_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight: dict[int, queue.Queue] = {}
        self._id_counter = itertools.count(1)
        self._closed = False

        self._stderr_buf: deque[str] = deque(maxlen=200)

        def _stderr_reader():
            if self.proc.stderr is None:
                raise ValueError("self.proc.stderr is None")
            for line in self.proc.stderr:
                self._stderr_buf.append(line.rstrip())

        self._stderr_thr = threading.Thread(target=_stderr_reader, daemon=True)
        self._stderr_thr.start()

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        atexit.register(self.close)

    def _read_loop(self):
        if not self.proc.stdout:
            raise ValueError("self.proc.stdout is None")
        for line in self.proc.stdout:
            # print(f"DEBUG: got line: {line!r}", flush=True)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = msg.get("id", None)
            type_ = msg.get("type", None)
            if type_ != "fnnx_stdio_response" or rid is None:
                continue

            with self._inflight_lock:
                q = self._inflight.pop(rid, None)
            if q is not None:
                q.put(msg)

    def _ensure_alive(self):
        code = self.proc.poll()
        if code is not None:
            err = "\n".join(self._stderr_buf) or "<no stderr>"
            raise RuntimeError(
                f"Server process exited with code {code}.\n--- server stderr ---\n{err}"
            )

    def request(self, handler, body, timeout=None):
        if self._closed:
            raise RuntimeError("Client is closed")
        if not isinstance(body, dict):
            raise TypeError("body must be a dict")

        self._ensure_alive()  # fail fast if server already died

        rid = next(self._id_counter)
        waitq: queue.Queue = queue.Queue(maxsize=1)
        with self._inflight_lock:
            self._inflight[rid] = waitq

        envelope = {"id": rid, "handler": handler, "body": body}
        line = json.dumps(envelope, allow_nan=False) + "\n"

        try:
            with self._write_lock:
                if not self.proc.stdin:
                    raise ValueError("self.proc.stdin is None")
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
        except BrokenPipeError:
            self._ensure_alive()
            raise

        try:
            response_envelope = (
                waitq.get(timeout=timeout) if timeout is not None else waitq.get()
            )
        except queue.Empty:
            with self._inflight_lock:
                self._inflight.pop(rid, None)
            raise TimeoutError(f"Timed out waiting for response to id={rid}")

        if response_envelope.get("status") == "error":
            raise RuntimeError(
                "The server returned the following error: "
                + response_envelope.get("error"),
            )

        return response_envelope.get("body", None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
