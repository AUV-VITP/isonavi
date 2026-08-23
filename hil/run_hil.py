"""Orchestrate a real hardware-in-the-loop run.

Runs entirely from WSL. The host plant listens on a local port. An SSH reverse
tunnel maps the board's localhost:PORT back to that listener, so the board
dials its own localhost and the traffic arrives at the host with no dependence
on Windows firewall rules or the USB-net routing direction. The flight computer
is launched on the board over the same SSH session.

Topology:

    board:  flight_computer  --connect-->  localhost:PORT  (board)
                                              | reverse tunnel (paramiko)
    WSL:    host_plant listens on          127.0.0.1:PORT  (WSL)
"""
import argparse
import json
import os
import select
import socket
import sys
import threading
import time

import paramiko

IP = "10.133.84.1"

# Resolved from this file rather than hardcoded, so the bench runs
# from any checkout.
_HIL = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HIL)
for _p in (_HIL, os.path.join(_HIL, "host"),
           os.path.join(_HIL, "common"),
           os.path.join(_REPO, "simulation")):
    sys.path.insert(0, _p)


# Reverse tunnel destinations, keyed by the port the board connects to.
ROUTES = {}


def _router(chan, origin, server):
    """Single dispatcher for every reverse tunnel on this transport."""
    dest = ROUTES.get(server[1])
    if dest is None:
        chan.close()
        return
    threading.Thread(target=_splice, args=(chan, dest[0], dest[1]),
                     daemon=True).start()


def reverse_forward(transport, remote_port, local_host, local_port):
    """Publish remote_port on the board, forwarded to local_host:local_port.

    Registering the router every time is harmless: paramiko keeps only one
    handler, and it is the same one.
    """
    ROUTES[remote_port] = (local_host, local_port)
    transport.request_port_forward("127.0.0.1", remote_port, handler=_router)


def _splice(chan, host, port):
    try:
        sock = socket.create_connection((host, port), timeout=10)
    except Exception as e:
        print(f"[tunnel] cannot reach host plant: {e}")
        chan.close()
        return
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    chan.setblocking(False)
    sock.setblocking(False)
    try:
        while True:
            r, _, _ = select.select([chan, sock], [], [], 1.0)
            if chan in r:
                d = chan.recv(65536)
                if not d:
                    break
                sock.sendall(d)
            if sock in r:
                d = sock.recv(65536)
                if not d:
                    break
                chan.sendall(d)
    except Exception:
        pass
    finally:
        chan.close()
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--max-time", type=float, default=1000.0)
    ap.add_argument("--out",
                    default=os.path.join(_HIL, "results",
                                         "hil_run.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--esp-bridge", default="",
                    help="host:port of the Windows side ESP32 bridge")
    ap.add_argument("--esp-port", type=int, default=5558,
                    help="port the board sees the bridge on, via the tunnel")
    args = ap.parse_args()

    import host_plant

    # 1. Start the host plant listener in a thread.
    result = {}

    def run_host():
        result["host"] = host_plant.serve(args.port, args.max_time, args.out, args.seed)

    hth = threading.Thread(target=run_host, daemon=True)
    hth.start()
    time.sleep(1.0)

    # 2. SSH to the board, set up the reverse tunnel, launch the flight computer.
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(IP, username="root", password="root", timeout=10,
                allow_agent=False, look_for_keys=False)
    tr = ssh.get_transport()
    tr.set_keepalive(5)

    stop = threading.Event()
    reverse_forward(tr, args.port, "127.0.0.1", args.port)

    if args.esp_bridge:
        bh, bp = args.esp_bridge.split(":")
        reverse_forward(tr, args.esp_port, bh, int(bp))
        print(f"[esp32] board localhost:{args.esp_port} tunnelled to "
              f"the bridge at {bh}:{bp}")
    time.sleep(0.5)

    if args.esp_bridge:
        esp_arg = f"--esp tcp:127.0.0.1:{args.esp_port}"
    else:
        esp_arg = "--esp ''"
    cmd = (f"cd /root/hil && python3 flight_computer.py "
           f"--host 127.0.0.1 --port {args.port} {esp_arg} "
           f"--max-time {args.max_time}")
    print(f"[board] launching: {cmd}")
    _, out, err = ssh.exec_command(cmd, timeout=args.max_time + 60)

    board_report = None
    for line in iter(out.readline, ""):
        line = line.rstrip()
        if line.startswith("REPORT "):
            board_report = json.loads(line[7:])
            print("[board] " + json.dumps(board_report, indent=1))
        elif line:
            print("[board] " + line)
    e = err.read().decode()
    if e.strip():
        print("[board stderr]\n" + e)

    stop.set()
    hth.join(timeout=30)
    ssh.close()

    print()
    print("=" * 60)
    print("HARDWARE-IN-THE-LOOP RUN COMPLETE")
    print("=" * 60)
    h = result.get("host", {})
    if board_report:
        print(f"  board loop timing (on RISC-V hardware):")
        print(f"    mean {board_report['loop_ms_mean']:.3f} ms, "
              f"p99 {board_report['loop_ms_p99']:.3f} ms, "
              f"max {board_report['loop_ms_max']:.3f} ms, "
              f"budget {board_report['budget_ms']:.0f} ms")
        rt = board_report["budget_ms"] / max(board_report["loop_ms_p99"], 1e-9)
        print(f"    real-time margin at p99: {rt:.1f}x")
        if board_report:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out.replace(".npz", "_board.json"), "w") as f:
                json.dump(board_report, f, indent=1)
    if board_report and board_report.get("esp_link", "none") != "none":
        sent = board_report.get("esp_pwm_sent", 0)
        ech = board_report.get("esp_echoes", 0)
        mis = board_report.get("esp_mismatches", 0)
        print(f"  actuator interface ({board_report['esp_link']}):")
        print(f"    pwm frames commanded  {sent}")
        print(f"    echoes returned       {ech}"
              f"  ({100.0 * ech / max(sent, 1):.1f} percent)")
        print(f"    width mismatches      {mis}")
        print(f"    esp crc errors        "
              f"{board_report.get('esp_crc_errors', 0)}")
    print(f"  host nav error: mean {h.get('nav_error_mean', 0):.3f} m, "
          f"max {h.get('nav_error_max', 0):.3f} m")
    print(f"  final phase: {h.get('final_phase', -1)} (7 = DONE)")
    print(f"  crc errors: {h.get('crc_errors', 0)}")


if __name__ == "__main__":
    main()
