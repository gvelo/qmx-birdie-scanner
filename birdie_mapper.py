#!/usr/bin/env python3
import serial
import numpy as np
import sounddevice as sd
import time
import h5py
import os
from datetime import datetime

# Symbolic variables
START_FREQUENCY = 28003300  # 28 MHz in Hz
END_FREQUENCY = 28003400    # 28.001 MHz in Hz (limited range for testing)
FREQUENCY_STEP = 10         # 10 Hz step
SETTLE_TIME = 0.150         # 150ms settle time
AUDIO_DEVICE_NAME = "QMX"
SERIAL_PORT_NAME = "/dev/ttyACM0"
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

        # Convert to dB scale (normalized)
        fft_magnitude = np.abs(fft_result)
        max_magnitude = np.max(fft_magnitude) if np.max(
            fft_magnitude) > 0 else 1
        fft_db = 20 * np.log10(fft_magnitude / max_magnitude)

        # Return only the filtered bins
        return fft_db[self.audio_passband_bins]


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "scan_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.join(output_dir, f"scan_{timestamp}.h5")

    # Initialize CAT control
    cat_control = CATControl(SERIAL_PORT_NAME)

    # Initialize audio
    audio = Audio(AUDIO_DEVICE_NAME)

    # Use the calculated FFT output size
    fft_length = audio.fft_output_size

    print(
        f"Starting frequency scan from {START_FREQUENCY/1000000:.6f} MHz to {END_FREQUENCY/1000000:.6f} MHz")
    print(
        f"Step size: {FREQUENCY_STEP} Hz, Settle time: {SETTLE_TIME*1000:.0f} ms")
    print(
        f"Total steps: {(END_FREQUENCY - START_FREQUENCY) // FREQUENCY_STEP}")
    print(f"FFT size: {fft_length} points")
    print(f"Output file: {h5_filename}")
    print("Press Ctrl+C to stop the scan")
    print("=" * 50)

    # Calculate total number of steps
    total_steps = (END_FREQUENCY - START_FREQUENCY) // FREQUENCY_STEP + 1

    # Initialize counters
    step_count = 0
    current_freq = START_FREQUENCY

    try:
        # Open HDF5 file for writing
        with h5py.File(h5_filename, 'w') as h5file:
            # Store metadata as attributes
            h5file.attrs['start_frequency'] = START_FREQUENCY
            h5file.attrs['end_frequency'] = END_FREQUENCY
            h5file.attrs['frequency_step'] = FREQUENCY_STEP
            h5file.attrs['sample_rate'] = audio.sample_rate
            h5file.attrs['fft_size'] = audio.fft_size
            h5file.attrs['timestamp'] = timestamp

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
            for freq in range(START_FREQUENCY, END_FREQUENCY + 1, FREQUENCY_STEP):
                # Set the radio frequency
                cat_control.setFreq(freq)

                # Wait for the radio to settle
                time.sleep(SETTLE_TIME)

                # Capture FFT at this frequency
                fft_db = audio.capture_and_fft()

                # Calculate average power in the FFT
                avg_power = np.mean(fft_db)
                max_power = np.max(fft_db)

                # Resize dataset and add new data (streaming style)
                if step_count > 0:  # After first iteration, need to resize
                    fft_dataset.resize(step_count + 1, axis=0)

                # Store FFT data
                fft_dataset[step_count] = fft_db

                # Print progress
                print(
                    f"Freq: {freq/1000000:.6f} MHz | Avg power: {avg_power:.2f} dB | Max power: {max_power:.2f} dB | Progress: {step_count+1}/{total_steps}")

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
        f"Scanned from {START_FREQUENCY/1000000:.6f} MHz to {current_freq/1000000:.6f} MHz")
    print(f"Total FFT captures: {step_count}")
