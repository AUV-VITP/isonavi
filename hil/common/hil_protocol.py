"""Wire protocol for the hardware-in-the-loop bench.

Three processors run one control loop between them:

    host (WSL)          physics, sonar, ground truth        the plant
    LicheeRV Nano       EKF, controller, mission FSM         the flight computer
    ESP32               8-channel ESC PWM                    the actuator

The board is the real-time master. Each tick it requests the current sensor
sample from the host, runs estimation and control, sends the resulting thrust
command to the ESP32, and returns the applied wrench to the host so the host
can integrate the plant forward. Nothing on the board ever sees ground truth,
exactly as in the pure-simulation stack, so a matching result is evidence that
the autonomy runs on hardware rather than that the hardware was told the
answer.

Framing is deliberately strict because a silently corrupted state estimate is
worse than a detected dropout: every frame is

    0xA5 0x5A | type(1) | length(2, LE) | payload(length) | crc16(2, LE)

CRC-16/CCITT-FALSE over type+length+payload. A bad CRC or lost sync is
reported, never acted on. Payloads are little-endian float64/float32/int, laid
out by struct format strings kept next to each message type so the host and
the board cannot disagree about the layout.
"""

from __future__ import annotations

import struct

SYNC0 = 0xA5
SYNC1 = 0x5A

# Message types. Host -> board and board -> host share one space; direction is
# implied by which side sends which type.
MSG_HELLO = 0x01          # board -> host: version, loop rate
MSG_SENSOR_REQ = 0x10     # board -> host: give me the sample at time t
MSG_SENSOR = 0x11         # host -> board: imu, dvl, depth for this tick
MSG_THRUST = 0x20         # board -> host: applied body wrench + per-thruster N
MSG_STATE = 0x21          # board -> host: current EKF pose estimate + phase
MSG_ACK = 0x2F            # either way: liveness
MSG_PWM = 0x30            # board -> ESP32: 8 pulse widths in microseconds
MSG_PWM_ECHO = 0x31       # ESP32 -> board: measured/applied pulse widths


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def encode(msg_type: int, payload: bytes) -> bytes:
    header = struct.pack("<BH", msg_type, len(payload))
    crc = crc16_ccitt(header + payload)
    return bytes([SYNC0, SYNC1]) + header + payload + struct.pack("<H", crc)


class FrameDecoder:
    """Incremental decoder. Feed bytes, get complete (type, payload) frames.

    Tolerant of partial reads and resynchronises after corruption by scanning
    for the sync pattern rather than giving up on the stream.
    """

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes):
        self.buf.extend(data)
        out = []
        while True:
            frame = self._try_one()
            if frame is None:
                break
            if frame is not False:
                out.append(frame)
        return out

    def _find_sync(self, start=0):
        b = self.buf
        i = start
        while i + 1 < len(b):
            if b[i] == SYNC0 and b[i + 1] == SYNC1:
                return i
            i += 1
        return -1

    def _try_one(self):
        b = self.buf
        i = self._find_sync()
        if i < 0:
            # No sync candidate; keep at most a trailing SYNC0.
            if len(b) > 1:
                del b[:-1]
            return None
        if i:
            del b[:i]
            self.resyncs += 1
        if len(b) < 5:
            return None
        msg_type = b[2]
        length = b[3] | (b[4] << 8)
        total = 5 + length + 2
        if len(b) < total:
            # Not enough bytes for the length this header claims. It may be a
            # false sync whose "length" is nonsense. If a genuine sync appears
            # later in what we already hold, this one was false: drop it and
            # rescan. Otherwise wait for more bytes.
            if self._find_sync(2) >= 0:
                del b[:2]
                self.resyncs += 1
                return False
            return None
        header = bytes(b[2:5])
        payload = bytes(b[5:5 + length])
        crc_rx = b[5 + length] | (b[6 + length] << 8)
        crc_calc = crc16_ccitt(header + payload)
        if crc_rx != crc_calc:
            # Bad CRC: drop only the sync bytes and rescan, so a real frame
            # beginning inside this bogus span is still recovered.
            self.crc_errors += 1
            del b[:2]
            self.resyncs += 1
            return False
        del b[:total]
        return (msg_type, payload)


# ---------------------------------------------------------------- payloads

# Sensor sample handed to the board each tick. The board must not receive the
# true pose; it gets only what real sensors would produce. A validity byte
# says which sensors actually reported this tick, so the board applies exactly
# the corrections a real system would (IMU 100 Hz, DVL 8 Hz, depth 20 Hz), and
# never a fabricated one. Bit 0 = imu, bit 1 = dvl, bit 2 = depth, bit 3 = dvl
# bottom lock.
#   t, valid(1), imu_gyro(3), imu_att(3), dvl_v(3), dvl_alt, depth
SENSOR_FMT = "<dB3d3d3ddd"
V_IMU = 1
V_DVL = 2
V_DEPTH = 4
V_LOCK = 8


def pack_sensor(t, valid, gyro, att, dvl_v, dvl_alt, depth):
    return struct.pack(SENSOR_FMT, t, int(valid), *gyro, *att, *dvl_v,
                       dvl_alt, depth)


def unpack_sensor(p):
    v = struct.unpack(SENSOR_FMT, p)
    valid = v[1]
    return {"t": v[0], "valid": valid,
            "has_imu": bool(valid & V_IMU), "has_dvl": bool(valid & V_DVL),
            "has_depth": bool(valid & V_DEPTH), "dvl_lock": bool(valid & V_LOCK),
            "gyro": v[2:5], "att": v[5:8], "dvl_v": v[8:11],
            "dvl_alt": v[11], "depth": v[12]}


# Sensor request: the tick time the board wants a sample for.
def pack_sensor_req(t):
    return struct.pack("<d", t)


def unpack_sensor_req(p):
    return struct.unpack("<d", p)[0]


# Applied thrust: body wrench and the 8 per-thruster forces the board
# allocated. The host integrates the plant with this wrench.
THRUST_FMT = "<6d8d"


def pack_thrust(wrench, thrusters):
    return struct.pack(THRUST_FMT, *wrench, *thrusters)


def unpack_thrust(p):
    v = struct.unpack(THRUST_FMT, p)
    return {"wrench": v[0:6], "thrusters": v[6:14]}


# State report: the board's own pose estimate, its sigma, the mission phase id
# and tick number. For telemetry and the equivalence comparison.
STATE_FMT = "<3d3d d i i"


def pack_state(pos, att, sigma, phase_id, tick):
    return struct.pack(STATE_FMT, *pos, *att, sigma, int(phase_id), int(tick))


def unpack_state(p):
    v = struct.unpack(STATE_FMT, p)
    return {"pos": v[0:3], "att": v[3:6], "sigma": v[6], "phase_id": v[7],
            "tick": v[8]}


# PWM command to the ESP32: 8 pulse widths in microseconds (uint16 each).
def pack_pwm(widths_us):
    assert len(widths_us) == 8
    return struct.pack("<8H", *[int(max(0, min(65535, w))) for w in widths_us])


def unpack_pwm(p):
    return list(struct.unpack("<8H", p))


PHASES = ("DEPLOY", "ACQUIRE", "TRANSIT", "SEARCH", "INSPECT", "RETURN",
          "REPORT", "DONE")
PHASE_ID = {name: i for i, name in enumerate(PHASES)}
