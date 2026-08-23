"""Start the ESP32 actuator loop through the bridge, then leave it running.

The bridge is a transparent pipe, so the MicroPython raw REPL is reachable
across it exactly as it is over the wire. Arming is kept as a separate step
rather than an autostart on boot, because a loop that owns the console from
power-on removes the recovery path; this way a plain reset always lands back
at a REPL.
"""

import argparse
import socket
import struct
import time

SYNC0, SYNC1 = 0xA5, 0x5A
MSG_PWM_ECHO = 0x31


def crc16(d, c=0xFFFF):
    for b in d:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 \
                else (c << 1) & 0xFFFF
    return c


def read_for(s, seconds):
    buf = bytearray()
    t0 = time.time()
    s.settimeout(0.05)
    while time.time() - t0 < seconds:
        try:
            d = s.recv(4096)
            if not d:
                break
            buf.extend(d)
        except socket.timeout:
            pass
        except Exception:
            break
    return bytes(buf)


def encode(mt, pl):
    h = struct.pack("<BH", mt, len(pl))
    return bytes([SYNC0, SYNC1]) + h + pl + struct.pack("<H", crc16(h + pl))


def find_echo(buf):
    i = buf.find(bytes([SYNC0, SYNC1]))
    while i >= 0 and len(buf) >= i + 23:
        if buf[i + 2] == MSG_PWM_ECHO:
            return list(struct.unpack("<8H", bytes(buf[i + 5:i + 21])))
        i = buf.find(bytes([SYNC0, SYNC1]), i + 1)
    return None


def arm(host, port, timeout=12.0):
    s = socket.create_connection((host, port), timeout=6)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Arming is idempotent. A loop left running from an earlier run has Ctrl-C
    # disabled, so probing it is the only way to tell it apart from a board
    # sitting at a REPL; a neutral command is harmless either way.
    s.sendall(encode(0x30, struct.pack("<8H", *([1500] * 8))))
    probe = bytearray(read_for(s, 1.2))
    already = find_echo(probe)
    if already is not None:
        print(f"[arm] actuator already running, channels at {already} us")
        s.close()
        return True

    # Interrupt anything running, enter the raw REPL, start the loop.
    s.sendall(b"\x03\x03")
    read_for(s, 0.4)
    s.sendall(b"\x01")
    out = read_for(s, 0.8)
    if b"raw REPL" not in out:
        print("[arm] no raw REPL prompt; is the board in a running loop?")
    s.sendall(b"import actuator\nactuator.run()\x04")

    # The loop announces itself with one echo frame carrying the armed state.
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout:
        buf.extend(read_for(s, 0.3))
        i = buf.find(bytes([SYNC0, SYNC1]))
        if i >= 0 and len(buf) >= i + 23:
            f = buf[i:i + 23]
            if f[2] == MSG_PWM_ECHO:
                widths = struct.unpack("<8H", bytes(f[5:21]))
                print(f"[arm] actuator armed, channels at {list(widths)} us")
                s.close()
                return True
    print("[arm] no armed frame seen")
    s.close()
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5558)
    a = ap.parse_args()
    raise SystemExit(0 if arm(a.host, a.port) else 1)
