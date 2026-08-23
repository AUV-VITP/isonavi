"""Minimal, reliable MicroPython raw-REPL client.

The friendly REPL echoes everything it receives and handles indented blocks by
guesswork, which makes pasting a multi-line program into it unreliable. That
cost me one wrong diagnosis already: a probe loop that never ran looked exactly
like a loop that ran and produced nothing.

Raw mode takes a whole program, executes it, and returns stdout and any
traceback separated by control bytes, with no echo at all.
"""

import time

import serial

PORT = "COM10"
BAUD = 115200


def open_port(port=PORT, baud=BAUD, reset=True, timeout=0.4):
    p = serial.Serial(port, baud, timeout=timeout)
    if reset:
        p.dtr = False
        p.rts = True
        time.sleep(0.15)
        p.rts = False
        time.sleep(1.7)
    p.reset_input_buffer()
    return p


def _read_until(p, token, timeout=6.0):
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = p.read(512)
        if d:
            buf.extend(d)
            if token in buf:
                return bytes(buf)
    return bytes(buf)


def enter_raw(p):
    p.write(b"\x03\x03")            # interrupt anything running
    time.sleep(0.3)
    p.read(65536)
    p.write(b"\x01")                # ctrl-A: raw REPL
    out = _read_until(p, b"raw REPL")
    p.read(65536)
    return b"raw REPL" in out


def exec_(p, code, timeout=10.0):
    """Run a program in raw mode. Returns (stdout, stderr)."""
    p.write(code.encode() + b"\x04")
    out = _read_until(p, b"\x04>", timeout)
    if out.startswith(b"OK"):
        out = out[2:]
    body = out.split(b"\x04>")[0]
    parts = body.split(b"\x04")
    stdout = parts[0].decode(errors="replace") if parts else ""
    stderr = parts[1].decode(errors="replace") if len(parts) > 1 else ""
    return stdout.strip(), stderr.strip()


def exit_raw(p):
    p.write(b"\x02")                # ctrl-B: back to the friendly REPL
    time.sleep(0.3)
    p.read(65536)


def put_file(p, local, remote, chunk=384):
    """Copy a local file to the device, base64 through raw mode."""
    import base64
    data = open(local, "rb").read()
    b64 = base64.b64encode(data).decode()
    exec_(p, "import ubinascii\nf=open('%s','wb')" % remote)
    for i in range(0, len(b64), chunk):
        out, err = exec_(p, "f.write(ubinascii.a2b_base64('%s'))"
                         % b64[i:i + chunk])
        if err:
            raise RuntimeError(err)
    exec_(p, "f.close()")
    out, err = exec_(p, "import os; print(os.stat('%s')[6])" % remote)
    return int(out.strip()) if out.strip().isdigit() else -1
