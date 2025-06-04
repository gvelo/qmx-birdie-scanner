#!/usr/bin/env python3
import serial

# Symbolic variables
START_FREQUENCY = 3500000  # 3.5 MHz in Hz
END_FREQUENCY = 30000000   # 30 MHz in Hz
AUDIO_DEVICE_NAME = "default"
SERIAL_PORT_NAME = "/dev/ttyACM0"

class CATControl:
    def __init__(self, port_name):
        self.port_name = port_name
        self.serial = serial.Serial(
            port=port_name,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
    
    def setFreq(self, frequency):
        command = f"FA{frequency};".encode()
        self.serial.write(command)

class Audio:
    def __init__(self, device_name):
        self.device_name = device_name

if __name__ == "__main__":
    cat_control = CATControl(SERIAL_PORT_NAME)
    cat_control.setFreq(28000100)