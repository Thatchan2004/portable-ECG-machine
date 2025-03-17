import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import pywt

# Hardware setup
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    chan = AnalogIn(ads, ADS.P0)
except Exception as e:
    print(f"Error initializing hardware: {e}")
    exit()

# Parameters
duration = 10  # Capture duration in seconds
sampling_rate = 250  # Samples per second
num_samples = duration * sampling_rate

# Data storage
time_data = np.zeros(num_samples)
ecg_data = np.zeros(num_samples)

# Filtering functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=5)
    y = lfilter(b, a, data)
    return y

# P-wave detection using wavelet transform
def detect_p_waves(ecg, time_data, sampling_rate):
    waveletname = 'db4'  # Daubechies 4 wavelet
    scales = np.arange(1, 128)  # Scales for wavelet transform
    coefficients, _ = pywt.cwt(ecg, scales, waveletname, sampling_period=1.0 / sampling_rate)

    # Find peaks in the wavelet coefficients (adjust threshold as needed)
    peak_indices, _ = find_peaks(np.abs(coefficients[30, :]), distance=50, height=0.5 * np.max(np.abs(coefficients[30, :])))

    p_wave_times = time_data[peak_indices]
    return p_wave_times

# Real-time data acquisition and plotting
plt.ion()  # Interactive plotting
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], color='black', linewidth=1, label='ECG signal')
expert_lines = []
basic_lines = []
proposed_lines = []
gray_bars = []

ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [mV]')
ax.set_xlim(18, 22)  # Set x-axis limits to match the example image
ax.set_ylim(-1.5, 1.5)
ax.legend()

start_time = time.time()
for i in range(num_samples):
    current_time = time.time() - start_time
    time_data[i] = current_time
    ecg_data[i] = chan.value * 4.096 / 32767
    time.sleep(1 / sampling_rate)

    if i % 50 == 0:  # Update plot every 50 samples
        filtered_ecg = butter_bandpass_filter(ecg_data[:i], 0.5, 40, sampling_rate)
        line.set_data(time_data[:i], filtered_ecg)

        # Basic P-wave detection (using peaks)
        peaks, _ = find_peaks(filtered_ecg, distance=150)
        basic_p_locations = time_data[:i][peaks]

        # Proposed P-wave detection (using wavelet)
        proposed_p_locations = detect_p_waves(filtered_ecg, time_data[:i], sampling_rate)

        # Example expert annotations (replace with your actual data)
        expert_p_locations = [18.5, 19.5, 20.5, 21.5]  # Example time locations

        # Example P-wave durations (replace with your actual data)
        p_wave_durations = [0.1, 0.12, 0.11, 0.13]  # Example durations

        # Update expert annotations
        for l in expert_lines:
            l.remove()
        expert_lines = []
        for loc in expert_p_locations:
            l = ax.axvline(x=loc, color='blue', linestyle='--', linewidth=1)
            expert_lines.append(l)

        # Update basic P-wave detections
        for l in basic_lines:
            l.remove()
        basic_lines = []
        for loc in basic_p_locations:
            l = ax.plot(loc, filtered_ecg[np.argmin(np.abs(time_data[:i] - loc))], 'x', color='blue', markersize=8)[0]
            basic_lines.append(l)

        # Update proposed P-wave detections
        for l in proposed_lines:
            l.remove()
        proposed_lines = []
        for loc in proposed_p_locations:
            l = ax.plot(loc, filtered_ecg[np.argmin(np.abs(time_data[:i] - loc))], 'o', color='red', markersize=8)[0]
            proposed_lines.append(l)

        # Update gray bars
        for bar in gray_bars:
            bar.remove()
        gray_bars = []
        for j, loc in enumerate(proposed_p_locations):
            start_time = loc - p_wave_durations[j] / 2
            end_time = loc + p_wave_durations[j] / 2
            bar = ax.axhspan(ymin=-1, ymax=-0.8, xmin=(start_time - 18) / 4, xmax=(end_time - 18) / 4, facecolor='gray', alpha=0.5)
            gray_bars.append(bar)

        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()
plt.show()
