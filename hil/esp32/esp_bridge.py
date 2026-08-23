"""TCP to serial bridge for the ESP32 actuator interface.

The flight computer, the plant and the ESP32 do not all live on the same
machine on this bench: the board is a USB network device, the plant runs under
WSL, and the ESP32 enumerates as a COM port on Windows, which WSL cannot open
without usbipd. Rather than move the plant or ask for a driver install, the
serial port is published on a socket and the frames pass through untouched.

Nothing here parses the protocol. Bytes in one side come out the other, so the
frames the ESP32 receives are byte for byte the ones the flight computer's
allocation produced.
"""

import argparse
import socket
import threading
import time

import serial


def pump(src_read, dst_write, name, stats):
    while True:
        try:
            data = src_read()
        except Exception:
            return
        if data is None:
            return
        if data:
            try:
                dst_write(data)
            except Exception:
                return
            stats[name] += len(data)
        else:
            time.sleep(0.001)


def serve(port, dev, baud):
    ser = serial.Serial(dev, baud, timeout=0)
    print(f"[bridge] {dev} at {baud} open")
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[bridge] listening on 0.0.0.0:{port}")

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(0.02)
        print(f"[bridge] client {addr}")
        stats = {"to_esp": 0, "from_esp": 0}
        ser.reset_input_buffer()

        def sock_read():
            try:
                d = conn.recv(4096)
                return None if d == b"" else d
            except socket.timeout:
                return b""
            except Exception:
                return None

        t = threading.Thread(target=pump,
                             args=(sock_read, ser.write, "to_esp", stats),
                             daemon=True)
        t.start()
        try:
            while t.is_alive():
                n = ser.in_waiting
                if n:
                    d = ser.read(n)
                    conn.sendall(d)
                    stats["from_esp"] += len(d)
                else:
                    time.sleep(0.001)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print(f"[bridge] closed, {stats['to_esp']} bytes to the ESP32, "
              f"{stats['from_esp']} back")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--dev", default="COM10")
    ap.add_argument("--baud", type=int, default=115200)
    a = ap.parse_args()
    serve(a.port, a.dev, a.baud)
