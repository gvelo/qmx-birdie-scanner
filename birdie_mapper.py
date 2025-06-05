#!/usr/bin/env python3
import serial
import numpy as np
import sounddevice as sd

# Symbolic variables
START_FREQUENCY = 28000000  # 28 MHz in Hz
END_FREQUENCY = 30000000   # 30 MHz in Hz
AUDIO_DEVICE_NAME = "QMX"
SERIAL_PORT_NAME = "/dev/ttyACM0"


class CATControl:
    """
    Class for controlling radio devices via CAT (Computer Aided Transceiver) interface.
    Handles serial communication with the radio to set frequencies and other parameters.
    """

    def __init__(self, port_name):
        """
        Initialize the CAT control interface with the specified serial port.

        Args:
            port_name (str): The name of the serial port to connect to (e.g., '/dev/ttyACM0')
        """
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
        """
        Set the frequency of the radio device.

        Args:
            frequency (int): The frequency in Hz to set the radio to
        """
        command = f"FA{frequency};".encode()
        self.serial.write(command)


class Audio:
    """
    Class for capturing and processing audio data from a sound device.
    Used for spectral analysis in birdie detection.
    """

    def __init__(self, device_name):
        """
        Initialize the audio capture interface with the specified device.
        Finds the device index by matching the device_name in the available devices.

        Args:
            device_name (str): The name of the audio device to use for capture

        Raises:
            ValueError: If the specified audio device cannot be found
        """
        self.device_name = device_name
        self.sample_rate = 48000
        self.fft_size = 8192

        # Get all available audio devices
        devices = sd.query_devices()

        # Find the device by name
        self.device_index = None
        for i, device in enumerate(devices):
            if device_name.lower() in device['name'].lower():
                self.device_index = i
                print(f"Found audio device: {device['name']} (index: {i})")
                break

        # Fail immediately if device not found
        if self.device_index is None:
            print(
                f"Error: Audio device '{device_name}' not found. Available devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']}")
            raise ValueError(f"Audio device '{device_name}' not found")

    def capture_and_fft(self):
        """
        Capture audio samples and compute the real FFT with Hann window.

        Returns:
            tuple: (frequencies, fft_db) where:
                frequencies (numpy.ndarray): Array of frequency bins
                fft_db (numpy.ndarray): FFT magnitude in dB scale
        """
        # Capture audio samples
        samples = sd.rec(self.fft_size, samplerate=self.sample_rate,
                         channels=1, dtype='float32', device=self.device_index)
        sd.wait()  # Wait until recording is finished

        # Extract the mono channel and apply Hann window
        samples = samples.flatten()
        window = np.hanning(len(samples))
        windowed_samples = samples * window

        # Compute real FFT
        fft_result = np.fft.rfft(windowed_samples)

        # Convert to dB scale (normalized)
        fft_magnitude = np.abs(fft_result)
        max_magnitude = np.max(fft_magnitude) if np.max(
            fft_magnitude) > 0 else 1
        fft_db = 20 * np.log10(fft_magnitude / max_magnitude)

        # Calculate frequency bins
        frequencies = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)

        return frequencies, fft_db


if __name__ == "__main__":
    # Initialize CAT control and set frequency
    cat_control = CATControl(SERIAL_PORT_NAME)
    cat_control.setFreq(28000100)

    # Initialize audio and capture FFT
    audio = Audio(AUDIO_DEVICE_NAME)
    frequencies, fft_db = audio.capture_and_fft()

    # Print FFT information
    print(f"FFT length: {len(fft_db)}")
    print(
        f"Frequency range: {frequencies[0]:.1f} Hz to {frequencies[-1]:.1f} Hz")
    print(
        f"Frequency resolution: {frequencies[1] - frequencies[0]:.1f} Hz")
    print("First 10 frequency bins:", frequencies[:10])
    
