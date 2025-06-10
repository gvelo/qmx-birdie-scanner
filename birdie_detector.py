#!/usr/bin/env python3
import h5py
import numpy as np
import os
import argparse
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import traceback


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

    print(f"    Plot saved: {filepath}")
    return filepath


def analyze_scan(file_path, output_dir=".", fft_plot=False):
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
                current_fft = fft_data[i]
                center_freq = scanned_frequencies[i]

                audio_bin_freqs = fft_bin_freqs[audio_bin_start:audio_bin_end+1]

                # Calculate noise floor using median
                noise_floor = np.median(current_fft)

                # Find peaks with in the FFT data
                birdie_peaks, _ = find_peaks(
                    current_fft,
                    prominence=40,
                    width=(1, 10),
                    distance=5
                )

                if len(birdie_peaks) > 0:
                    print(
                        f"Center Freq: {center_freq/1000000:.6f} MHz | Noise Floor: {noise_floor:.1f} dB")
                    print(f"  Detected birdies ({len(birdie_peaks)}):")

                    for peak_idx in birdie_peaks:
                        freq_offset = audio_bin_freqs[peak_idx]
                        print(f"    {freq_offset:.1f} Hz")

                    save_fft_plot(current_fft, center_freq,
                                  audio_bin_freqs, birdie_peaks, noise_floor, output_dir)
                    print()
                else:
                    # avg_power = np.mean(current_fft)
                    # print(f"Center Freq: {center_freq/1000000:.6f} MHz | Noise Floor: {noise_floor:.1f} dB | No birdies detected | Avg: {avg_power:.1f} dB")

                    if fft_plot:
                        save_fft_plot(current_fft, center_freq,
                                      audio_bin_freqs, [], noise_floor, output_dir)

    except Exception as e:
        print(f"Error analyzing file: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='QMX Birdie Detection Analysis')
    parser.add_argument('h5_file', help='Input H5 file with FFT data to analyze')
    parser.add_argument('--output-dir', default='.', 
                       help='Output directory for FFT plots (default: current directory)')
    parser.add_argument('--fft-plot', action='store_true',
                       help='Generate plots for ALL FFTs (WARNING: debugging only, creates many files)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Starting FFT data analysis")
    print(f"Input file: {args.h5_file}")
    print(f"Output directory: {args.output_dir}")
    if args.fft_plot:
        print("WARNING: FFT plotting enabled for ALL frequencies - this will generate many files!")
    print("=" * 50)
    
    analyze_scan(args.h5_file, args.output_dir, args.fft_plot)


if __name__ == "__main__":
    main()
