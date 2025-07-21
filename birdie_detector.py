#!/usr/bin/env python3
import h5py
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import argparse
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import traceback

# Birdie detection parameters
HEIGHT_THRESHOLD_DB = 19  # dB above noise floor for peak detection


def find_global_max_magnitude(h5file):
    """Find the global maximum magnitude across all FFT data without loading everything into memory."""
    fft_data = h5file['fft_data']
    fft_count = fft_data.shape[0]

    global_max = 0.0
    chunk_size = 1000  # Process 1000 FFTs at a time

    for i in range(0, fft_count, chunk_size):
        end_idx = min(i + chunk_size, fft_count)
        chunk = fft_data[i:end_idx]
        chunk_max = np.max(chunk)
        if chunk_max > global_max:
            global_max = chunk_max

    return global_max


def create_nonlinear_birdie_colormap():

    colors = [
        '#000022',
        '#000044',
        '#000066',
        '#002288',
        '#0044AA',
        '#0066CC',
        '#00AAEE',
        '#66DDFF',
        '#AAFFDD',
        '#DDFFAA',
        '#FFDD66',
        '#FFAA00',
        '#FF6600',
        '#FFFF00'
    ]

    breakpoints = [0.0, 0.60, 0.70, 0.75, 0.80, 0.83, 0.86, 0.89,
                   0.92, 0.94, 0.96, 0.98, 0.99, 1.0]

    birdie_cmap = LinearSegmentedColormap.from_list(
        'birdie_nonlinear',
        list(zip(breakpoints, colors)),
        N=256
    )

    return birdie_cmap


def plot_spectrogram(h5file, output_dir="."):
    """Generate and save spectrogram plot using globally normalized FFT data."""

    # Get metadata
    start_freq = h5file.attrs.get('start_frequency', 0)
    end_freq = h5file.attrs.get('end_frequency', 0)
    freq_step = h5file.attrs.get('frequency_step', 0)
    sample_rate = h5file.attrs.get('sample_rate', 0)
    fft_size = h5file.attrs.get('fft_size', 0)
    audio_bin_start = h5file.attrs.get('audio_bin_start', 0)
    audio_bin_end = h5file.attrs.get('audio_bin_end', 0)

    # Get FFT data
    fft_data = h5file['fft_data']
    fft_count = fft_data.shape[0]

    # Find global maximum for normalization
    print("Finding global maximum for normalization...")
    global_max_magnitude = find_global_max_magnitude(h5file)
    print(f"Global max magnitude: {global_max_magnitude:.2e}")

    # Create frequency arrays
    scanned_frequencies = np.arange(
        start_freq, end_freq + 1, freq_step)[:fft_count]
    fft_bin_freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
    audio_bin_freqs = fft_bin_freqs[audio_bin_start:audio_bin_end+1]

    # Create spectrogram plot
    plt.figure(figsize=(16, 10))

    # Create meshgrid for pcolormesh
    # X-axis: audio frequencies (150-3500 Hz)
    # Y-axis: reception frequencies
    # Convert to MHz
    X, Y = np.meshgrid(audio_bin_freqs, scanned_frequencies / 1e6)

    # Convert to dB with global normalization and plot
    print("Converting to dB and creating spectrogram...")
    fft_data_db = 20 * np.log10(fft_data[:] / global_max_magnitude)
    im = plt.pcolormesh(X, Y, fft_data_db, shading='auto',
                        cmap=create_nonlinear_birdie_colormap())
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)

    # Labels and formatting
    plt.xlabel('Audio Frequency (Hz)')
    plt.ylabel('Reception Frequency (MHz)')
    plt.title(f'Spectrogram: {start_freq/1e6:.6f} - {end_freq/1e6:.6f} MHz')
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = f"spectrogram_{start_freq/1e6:.6f}_{end_freq/1e6:.6f}MHz.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Spectrogram saved to: {filepath}")
    return filepath


def save_fft_plot(fft_data, center_freq, fft_bin_freqs, peaks, noise_floor, output_dir="."):
    """Generate and save FFT plot when birdies are detected."""

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(fft_bin_freqs, fft_data, 'b-', alpha=0.7, label='FFT Data')

    # Mark detected peaks
    if len(peaks) > 0:
        peak_freqs = fft_bin_freqs[peaks]
        peak_powers = fft_data[peaks]
        plt.plot(peak_freqs, peak_powers, 'ro',
                 markersize=8, label='Detected Birdies')

    # Add noise floor line
    plt.axhline(y=noise_floor, color='g', linestyle='--',
                alpha=0.7, label=f'Noise Floor: {noise_floor:.1f} dB')

    # Labels and formatting
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title(f'FFT Analysis - Center Freq: {center_freq/1000000:.6f} MHz')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot to specified output directory
    filename = f"fft_plot_{center_freq/1000000:.6f}MHz.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def analyze_scan(file_path, output_dir=".", plot_mode="none"):
    """Analyze a scan file and detect birdies"""

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Analyzing scan file: {file_path}")
    print("=" * 50)

    try:
        with h5py.File(file_path, 'r') as h5file:
            # Print metadata
            start_freq = h5file.attrs.get('start_frequency', 0)
            end_freq = h5file.attrs.get('end_frequency', 0)
            freq_step = h5file.attrs.get('frequency_step', 0)
            sample_rate = h5file.attrs.get('sample_rate', 0)
            fft_size = h5file.attrs.get('fft_size', 0)
            audio_bin_start = h5file.attrs.get('audio_bin_start', 0)
            audio_bin_end = h5file.attrs.get('audio_bin_end', 0)

            print(
                f"Frequency range: {start_freq/1000000:.6f} MHz to {end_freq/1000000:.6f} MHz")
            print(f"Frequency step: {freq_step} Hz")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"FFT size: {fft_size}")
            print(
                f"Audio filter bin range bins: {audio_bin_start} to {audio_bin_end}")

            # Get FFT data
            fft_data = h5file['fft_data']
            fft_count = fft_data.shape[0]
            fft_length = fft_data.shape[1]

            print(f"Total FFT captures: {fft_count}")
            print(f"FFT audio length: {fft_length} points")
            print("-" * 50)

            scanned_frequencies = np.arange(
                start_freq, end_freq + 1, freq_step)[:fft_count]

            # Calculate frequencies for FFT bins
            fft_bin_freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)

            # Analyze each FFT for birdies
            print("Birdie Analysis:")
            print("-" * 50)

            for i in range(fft_count):
                current_fft_raw = fft_data[i]
                center_freq = scanned_frequencies[i]

                audio_bin_freqs = fft_bin_freqs[audio_bin_start:audio_bin_end+1]

                # Convert raw magnitude to dB for birdie detection (local normalization)
                max_magnitude = np.max(current_fft_raw) if np.max(
                    current_fft_raw) > 0 else 1
                current_fft = 20 * np.log10(current_fft_raw / max_magnitude)

                # Calculate noise floor using median
                noise_floor = np.median(current_fft)

                # Find peaks with in the FFT data
                birdie_peaks, props = find_peaks(
                    current_fft,
                    prominence=20,
                    width=(1, 10),
                    distance=5,
                    wlen=50,
                    height=noise_floor + HEIGHT_THRESHOLD_DB
                )

                if len(birdie_peaks) > 0:
                    birdie_freqs = [
                        f"{audio_bin_freqs[peak_idx]:.1f}Hz" for peak_idx in birdie_peaks]
                    print(
                        f"Center Freq: {center_freq/1000000:.6f} MHz | Birdies: {', '.join(birdie_freqs)}")

                    # Generate plot for birdie or all modes
                    if plot_mode in ["birdie", "all"]:
                        save_fft_plot(current_fft, center_freq,
                                      audio_bin_freqs, birdie_peaks, noise_floor, output_dir)
                else:
                    # avg_power = np.mean(current_fft)
                    # print(f"Center Freq: {center_freq/1000000:.6f} MHz | Noise Floor: {noise_floor:.1f} dB | No birdies detected | Avg: {avg_power:.1f} dB")

                    # Generate plot only for all mode when no birdies detected
                    if plot_mode == "all":
                        save_fft_plot(current_fft, center_freq,
                                      audio_bin_freqs, [], noise_floor, output_dir)

    except Exception as e:
        print(f"Error analyzing file: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='QMX Birdie Detection Analysis')
    parser.add_argument(
        'h5_file', help='Input H5 file with FFT data to analyze')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for FFT plots (default: current directory)')
    parser.add_argument('--plot-mode', choices=['none', 'birdie', 'all', 'spectrogram'], default='none',
                        help='Plot generation mode: none=no plots, birdie=only frequencies with birdies, all=all frequencies (WARNING: creates many files), spectrogram=generate spectrogram')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Starting FFT data analysis")
    print(f"Input file: {args.h5_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plot mode: {args.plot_mode}")
    if args.plot_mode == "all":
        print("WARNING: Plot mode 'all' will generate many files!")
    print("=" * 50)

    # Handle spectrogram mode independently
    if args.plot_mode == "spectrogram":
        with h5py.File(args.h5_file, 'r') as h5file:
            plot_spectrogram(h5file, args.output_dir)
    else:
        analyze_scan(args.h5_file, args.output_dir, args.plot_mode)


if __name__ == "__main__":
    main()
