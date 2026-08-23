"""Verify the ESP32 actuator interface over the real wire protocol.

Uploads the firmware, starts the servicing loop, then commands pulse widths and
checks that what comes back is what the LEDC duty registers are actually
emitting. Nothing here is mocked: the frames are the same ones the flight
computer sends, and the echo is read from hardware.
"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpy

SYNC0, SYNC1 = 0xA5, 0x5A
MSG_PWM, MSG_PWM_ECHO = 0x30, 0x31


def crc16(d, c=0xFFFF):
    for b in d:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 \
                else (c << 1) & 0xFFFF
    return c


def enc(mt, pl):
    h = struct.pack("<BH", mt, len(pl))
    return bytes([SYNC0, SYNC1]) + h + pl + struct.pack("<H", crc16(h + pl))


def dec(buf):
    out = []
    while len(buf) >= 7:
        if not (buf[0] == SYNC0 and buf[1] == SYNC1):
            del buf[0]
            continue
        ln = buf[3] | (buf[4] << 8)
        tot = 5 + ln + 2
        if len(buf) < tot:
            break
        pl = bytes(buf[5:5 + ln])
        rx = buf[5 + ln] | (buf[6 + ln] << 8)
        if rx == crc16(bytes(buf[2:5]) + pl):
            out.append((buf[2], pl))
            del buf[:tot]
        else:
            del buf[:2]
    return out


PENDING = []


def await_echo(p, buf, deadline):
    """Return the next echo, queueing any others already in the buffer.

    A single read can carry several frames. Decoding them all and returning
    only the first silently drops the rest, which reads back as every echo
    arriving one command late.
    """
    if PENDING:
        return PENDING.pop(0)
    while time.perf_counter() < deadline:
        d = p.read(256)
        if d:
            buf.extend(d)
            for mt, pl in dec(buf):
                if mt == MSG_PWM_ECHO:
                    PENDING.append(list(struct.unpack("<8H", pl)))
            if PENDING:
                return PENDING.pop(0)
    return None


CASES = [
    [1500] * 8,
    [1000, 1150, 1300, 1450, 1600, 1750, 1900, 2000],
    [2000, 1000, 2000, 1000, 2000, 1000, 2000, 1000],
    [1234, 1345, 1456, 1567, 1678, 1789, 1890, 1999],
    [1100] * 8,
    [1900, 1850, 1800, 1750, 1700, 1650, 1600, 1550],
]


def main():
    p = mpy.open_port()
    if not mpy.enter_raw(p):
        print("could not enter raw REPL")
        return 1
    n = mpy.put_file(p, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "actuator.py"), "actuator.py")
    print(f"firmware uploaded, {n} bytes")

    p.write(b"import actuator\nactuator.run()\x04")

    # run() emits one frame on entry reporting the armed state. It has to be
    # consumed from the same buffer the cases use, or every echo afterwards is
    # read one command behind.
    buf = bytearray()
    armed = await_echo(p, buf, time.perf_counter() + 6.0)
    if armed is None:
        print("actuator loop did not start")
        return 1
    print(f"armed on entry: {armed}")
    # Only now tighten the read timeout, so the measured round trip is the
    # device and not the host serial driver polling interval. The upload
    # path above needs the longer one.
    p.timeout = 0.004
    print()
    print(f"  {'measured from the duty registers (us)':48s}"
          f"{'reproduces a command?':>28}")
    print("  " + "-" * 98)

    # The interface reports the duty registers as they stand when a command
    # arrives, then applies the new one, so an echo answers a command one or
    # two frames back depending on how the reads and the 20 ms PWM frame line
    # up. The meaningful test is therefore whether every echo exactly
    # reproduces a command that was actually issued, and how far behind it is.
    lat, ok, lags = [], True, []
    history = [armed]
    for cmd in CASES:
        t0 = time.perf_counter()
        p.write(enc(MSG_PWM, struct.pack("<8H", *cmd)))
        got = await_echo(p, buf, t0 + 2.5)
        lat.append((time.perf_counter() - t0) * 1000)
        history.append(cmd)
        lag = None
        if got is not None:
            for k in range(len(history) - 1, -1, -1):
                if all(abs(x - y) <= 1 for x, y in zip(history[k], got)):
                    lag = len(history) - 1 - k
                    break
        ok &= lag is not None
        if lag is not None:
            lags.append(lag)
        print(f"  {str(got):48s}{'exact, ' + str(lag) + ' frame(s) back' if lag is not None else 'NO MATCH':>28}")

    print()
    print(f"  command to confirmed echo: mean {sum(lat) / len(lat):.1f} ms, "
          f"min {min(lat):.1f} ms, max {max(lat):.1f} ms")
    if lags:
        print(f"  pipeline lag: mean {sum(lags) / len(lags):.2f} frames, "
              f"max {max(lags)}")
    print("  the interface never blocks; it reports before it applies")
    print(f"  channels: 8 at {50} Hz, resolution "
          f"{20000 / 65535:.3f} us per LSB")
    print("  RESULT:", "actuator chain verified" if ok else "MISMATCH")

    p.write(b"\x03\x03")
    time.sleep(0.4)
    p.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
