# Hardware in the loop

The autonomy stack running on real hardware, driven by the physics simulator.

## Roles

| Node | Hardware | Runs |
| --- | --- | --- |
| host | x86 (WSL) | physics, sonar, sensor models, ground truth |
| flight computer | LicheeRV Nano, RISC-V rv64 @ 750 MHz | EKF, controller, mission state machine |
| actuator | ESP32-D0WD-V3 | 8-channel ESC PWM |

The board is the real-time master. Each tick it requests the sensor sample
from the host, runs estimation and control, drives the ESP32, and returns the
applied wrench so the host integrates the plant. Ground truth never leaves the
host, so a matching mission result is evidence the autonomy runs on hardware
rather than that the hardware was handed the answer.

## Wire protocol

`common/hil_protocol.py`. Framed as `A5 5A | type | len | payload | crc16`,
CRC-16/CCITT-FALSE. The decoder resynchronises after corruption and rejects bad
frames rather than acting on them. 12 protocol tests in
`common/test_protocol.py`, including a 5000-frame lossy-channel stress test.

## Layout

- `common/`  protocol and its tests
- `host/`    host plant (physics + sensors + truth)
- `board/`   flight computer (deployed to the LicheeRV under /root/hil)
- `deploy_board.py`  SFTP the board code onto the LicheeRV
- `run_hil.py`  orchestrate a real run over an SSH reverse tunnel
- `test_hil_loopback.py`  full loop on localhost with no hardware, checks
  equivalence against the pure-simulation result

## Reproducing

```bash
python test_hil_loopback.py         # no hardware, equivalence check
python deploy_board.py              # push code to the board
python run_hil.py --max-time 60     # real board + host
```
