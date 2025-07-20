#!/usr/bin/env python3
import serial
import numpy as np
import sounddevice as sd
import time
import h5py
import os
import argparse
from datetime import datetime

# Configuration constants
FREQUENCY_STEP = 10         # 10 Hz step
SETTLE_TIME = 0.150         # 150ms settle time
AUDIO_DEVICE_NAME = "QMX"
SAMPLE_RATE = 48000         # 48 kHz sample rate
FFT_SIZE = 8192             # 8192 samples per FFT

# QMX audio passband filter settings
# These values are based on the QMX audio passband filter characteristics
AUDIO_BAND_START_HZ = 150         # Minimum frequency in Hz for bin filtering
AUDIO_BAND_END_HZ = 3500          # Maximum frequency in Hz for bin filtering


class CATControl:
    """
    Class for controlling radio devices via CAT interface.
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
    Class for capturing and processing audio data from QMX audio device. 
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
        self.sample_rate = SAMPLE_RATE
        self.fft_size = FFT_SIZE

        # Calculate the size of real FFT output (N/2 + 1)
        self.fft_output_size = (self.fft_size // 2) + 1

        # Calculate filtered FFT bin range
        self.audio_passband_bins = self.get_audio_passband_bins()

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

    def get_audio_passband_bins(self):
        """
        Calculate the FFT bin range that corresponds to the QMX audio passband filter.
        Returns:
            array of int: Indices of FFT bins that fall within the QMX audio passband filter range
        """

        # Calculate FFT bin frequencies
        fft_bin_freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)

        audio_passband_bins = np.where((fft_bin_freqs >= AUDIO_BAND_START_HZ) &
                                       (fft_bin_freqs <= AUDIO_BAND_END_HZ))[0]

        print(
            f"Audio passband bins count: {len(audio_passband_bins)} bins from {AUDIO_BAND_START_HZ}-{AUDIO_BAND_END_HZ} Hz range")
        print(
            f"Frequency range of filtered bins: {fft_bin_freqs[audio_passband_bins[0]]:.1f} - {fft_bin_freqs[audio_passband_bins[-1]]:.1f} Hz")

        return audio_passband_bins

    def capture_and_fft(self):
        """
        Capture audio samples and compute the real FFT with Hann window.
        Returns the FFT magnitude in dB scale, filtered to the QMX audio passband. 

        Returns:
            numpy.ndarray: Array of FFT magnitudes in dB scale, filtered to the QMX audio passband
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

        # Get magnitude and return raw values (no normalization)
        fft_magnitude = np.abs(fft_result)

        # Return only the filtered bins (raw magnitude values)
        return fft_magnitude[self.audio_passband_bins]


def main():
    parser = argparse.ArgumentParser(description='QMX FFT Frequency Scanner')
    parser.add_argument('start_freq', type=int, help='Start frequency in Hz')
    parser.add_argument('end_freq', type=int, help='End frequency in Hz')
    parser.add_argument('output_file', help='Output H5 filename')
    parser.add_argument('--serial-port', default='/dev/ttyACM0',
                        help='Serial port for CAT control (default: /dev/ttyACM0)')

    args = parser.parse_args()

    # Use command line arguments
    start_frequency = args.start_freq
    end_frequency = args.end_freq
    h5_filename = args.output_file
    serial_port = args.serial_port

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(h5_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize CAT control
    cat_control = CATControl(serial_port)

    # Initialize audio
    audio = Audio(AUDIO_DEVICE_NAME)

    # Use the calculated FFT output size
    fft_length = audio.fft_output_size

    print(
        f"Starting frequency scan from {start_frequency/1000000:.6f} MHz to {end_frequency/1000000:.6f} MHz")
    print(
        f"Step size: {FREQUENCY_STEP} Hz, Settle time: {SETTLE_TIME*1000:.0f} ms")
    print(
        f"Total steps: {(end_frequency - start_frequency) // FREQUENCY_STEP}")
    print(f"FFT size: {fft_length} points")
    print(f"Output file: {h5_filename}")
    print(f"Serial port: {serial_port}")
    print("Press Ctrl+C to stop the scan")
    print("=" * 50)

    # Calculate total number of steps
    total_steps = (end_frequency - start_frequency) // FREQUENCY_STEP + 1

    # Initialize counters
    step_count = 0
    current_freq = start_frequency

    try:
        # Open HDF5 file for writing
        with h5py.File(h5_filename, 'w') as h5file:
            # Store metadata as attributes
            h5file.attrs['start_frequency'] = start_frequency
            h5file.attrs['end_frequency'] = end_frequency
            h5file.attrs['frequency_step'] = FREQUENCY_STEP
            h5file.attrs['sample_rate'] = audio.sample_rate
            h5file.attrs['fft_size'] = audio.fft_size
            h5file.attrs['timestamp'] = datetime.now().strftime(
                "%Y%m%d_%H%M%S")

            # Add filtered frequency range attributes
            frequencies_fft = np.fft.rfftfreq(
                audio.fft_size, 1/audio.sample_rate)
            h5file.attrs['filtered_min_freq'] = frequencies_fft[audio.audio_passband_bins[0]]
            h5file.attrs['filtered_max_freq'] = frequencies_fft[audio.audio_passband_bins[-1]]
            h5file.attrs['audio_bin_start'] = audio.audio_passband_bins[0]
            h5file.attrs['audio_bin_end'] = audio.audio_passband_bins[-1]

            # Create resizable dataset for filtered FFT results
            fft_dataset = h5file.create_dataset(
                'fft_data',
                shape=(1, len(audio.audio_passband_bins)),
                maxshape=(None, len(audio.audio_passband_bins)),
                dtype='float32',
                chunks=True  # Enable chunking for efficient resizing
            )

            # Scan through frequency range
            for freq in range(start_frequency, end_frequency + 1, FREQUENCY_STEP):
                # Set the radio frequency
                cat_control.setFreq(freq)

                # Wait for the radio to settle
                time.sleep(SETTLE_TIME)

                # Capture FFT at this frequency
                fft_magnitude = audio.capture_and_fft()

                # Calculate average and max magnitude
                avg_magnitude = np.mean(fft_magnitude)
                max_magnitude = np.max(fft_magnitude)

                # Resize dataset and add new data (streaming style)
                if step_count > 0:  # After first iteration, need to resize
                    fft_dataset.resize(step_count + 1, axis=0)

                # Store FFT data
                fft_dataset[step_count] = fft_magnitude

                # Print progress
                print(
                    f"Freq: {freq/1000000:.6f} MHz | Avg magnitude: {avg_magnitude:.2e} | Max magnitude: {max_magnitude:.2e} | Progress: {step_count+1}/{total_steps}")

                # Increment counter
                step_count += 1
                current_freq = freq

                # Flush data to disk periodically
                if step_count % 100 == 0:
                    h5file.flush()

    except KeyboardInterrupt:
        print("\nScan interrupted by user")

    print("=" * 50)
    print(f"Scan complete. Data saved to {h5_filename}")
    print(
        f"Scanned from {start_frequency/1000000:.6f} MHz to {current_freq/1000000:.6f} MHz")
    print(f"Total FFT captures: {step_count}")


if __name__ == "__main__":
    main()
