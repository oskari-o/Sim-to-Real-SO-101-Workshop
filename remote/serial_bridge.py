#!/usr/bin/env python3
"""Forward a local serial port (e.g. the SO-101 leader arm on a Mac) over TCP.

Run on the local Mac with this script:
    python3 serial_bridge.py /dev/cu.usbmodemXXXX

It listens on 127.0.0.1:54321. Expose it to the GPU server with an SSH
reverse tunnel (ssh -N -R 54321:localhost:54321 upcloud_gpu), then on the
server create a virtual serial port that connects to it with socat.
"""
import argparse
import socket
import threading

import serial  # pip install pyserial (already present in a LeRobot env)


def pump_serial_to_net(ser, conn, stop):
    try:
        while not stop.is_set():
            data = ser.read(256)
            if data:
                conn.sendall(data)
    except (OSError, serial.SerialException):
        pass
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port", help="serial device, e.g. /dev/cu.usbmodem5A7A0185421")
    p.add_argument("--baud", type=int, default=1_000_000, help="Feetech bus baud (SO-101: 1000000)")
    p.add_argument("--listen", type=int, default=54321, help="local TCP port to listen on")
    args = p.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.001)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.listen))
    srv.listen(1)
    print(f"Bridging {args.port} @ {args.baud} baud on 127.0.0.1:{args.listen}. Ctrl-C to stop.")

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Client connected")
        ser.reset_input_buffer()
        stop = threading.Event()
        t = threading.Thread(target=pump_serial_to_net, args=(ser, conn, stop), daemon=True)
        t.start()
        try:
            while not stop.is_set():
                data = conn.recv(256)
                if not data:
                    break
                ser.write(data)
        except OSError:
            pass
        finally:
            stop.set()
            conn.close()
            t.join(timeout=1)
            print("Client disconnected, waiting for the next one")


if __name__ == "__main__":
    main()
