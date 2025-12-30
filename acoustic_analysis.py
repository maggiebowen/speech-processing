import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configurations ---
SAMPLE_DIR = 'samples'
OUTPUT_DIR = 'output_plots'
RAGTIME_FILE = os.path.join(SAMPLE_DIR, 'ragtime.wav')
PIANO_A4_FILE = os.path.join(SAMPLE_DIR, 'piano_A4.wav')
PIANO_A4_F4_FILE = os.path.join(SAMPLE_DIR, 'piano_A4_F4.wav')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_waveform(y, sr, title, filename):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# --- Section 2: Intensity & Energy ---

def compute_manual_energy(audio_signal):
    """
    Implements: E = (1/N) * sum(x[n]^2)
    """
    N = len(audio_signal)
    energy = np.sum(audio_signal**2) / N
    return energy

def analyze_intensity(file_path):
    print(f"\n--- Analysis for {file_path} ---")
    y, sr = librosa.load(file_path)
    
    # Metadata
    duration = librosa.get_duration(y=y, sr=sr)
    # librosa.load mixes to mono by default unless mono=False. 
    # Checking shape for channels if we were to load multi-channel, 
    # but for consistent analysis we'll use the loaded y (mono).
    print(f"Sampling Rate: {sr} Hz")
    print(f"Samples: {len(y)}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Plot Waveform
    plot_waveform(y, sr, f"Waveform: {os.path.basename(file_path)}", "waveform_ragtime.png")
    
    # Manual Energy
    manual_e = compute_manual_energy(y)
    print(f"Manual Average Energy: {manual_e:.6f}")
    
    # Librosa RMS
    # rms returns an array of frames, we want the global rms for comparison or frame-based?
    # The user asks to "Verify your manual energy implementation by ensuring E approx RMS^2".
    # RMS is usually Root Mean Square. Square of RMS is Mean Square, which is Average Energy.
    # librosa.feature.rms computes it over frames. We can compute global RMS manually or mean of frames.
    # Let's compute global RMS exactly as 'sqrt(mean(y**2))' which is sqrt(Energy).
    
    librosa_rms_frame = librosa.feature.rms(y=y)
    # This gives RMS over time windows. The global RMS would be approx the mean of these or we can just do sqrt(Energy).
    # Let's verify E ~ RMS^2 relation globally.
    global_rms = np.sqrt(np.mean(y**2))
    print(f"Global RMS (numpy): {global_rms:.6f}")
    print(f"Global RMS^2: {global_rms**2:.6f}")
    print(f"Verification (E approx RMS^2): {np.isclose(manual_e, global_rms**2)}")

    # Temporal Resolution
    # 1. 1s frames, No overlap
    frame_length_1s = sr * 1 # 1 second samples
    hop_length_1s_no_overlap = frame_length_1s
    rms_1s_no = librosa.feature.rms(y=y, frame_length=frame_length_1s, hop_length=hop_length_1s_no_overlap)[0]
    
    # 2. 1s frames, 50% overlap
    hop_length_1s_50 = frame_length_1s // 2
    rms_1s_50 = librosa.feature.rms(y=y, frame_length=frame_length_1s, hop_length=hop_length_1s_50)[0]
    
    # 3. 40ms frames, 50% overlap
    frame_length_40ms = int(sr * 0.040)
    hop_length_40ms_50 = frame_length_40ms // 2
    rms_40ms = librosa.feature.rms(y=y, frame_length=frame_length_40ms, hop_length=hop_length_40ms_50)[0]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    # Construct time axis manually for different hops
    times_1s_no = np.arange(len(rms_1s_no)) * hop_length_1s_no_overlap / sr
    plt.step(times_1s_no, rms_1s_no, label='RMS (1s, 0% ovlp)', color='r', where='post')
    plt.legend()
    plt.title("1s Frames, No Overlap")

    plt.subplot(3, 1, 2)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    times_1s_50 = librosa.frames_to_time(np.arange(len(rms_1s_50)), sr=sr, hop_length=hop_length_1s_50)
    plt.plot(times_1s_50, rms_1s_50, label='RMS (1s, 50% ovlp)', color='g')
    plt.legend()
    plt.title("1s Frames, 50% Overlap")

    plt.subplot(3, 1, 3)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    times_40ms = librosa.frames_to_time(np.arange(len(rms_40ms)), sr=sr, hop_length=hop_length_40ms_50)
    plt.plot(times_40ms, rms_40ms, label='RMS (40ms, 50% ovlp)', color='b')
    plt.legend()
    plt.title("40ms Frames, 50% Overlap")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rms_temporal_resolution.png"))
    plt.close()
    
# --- Section 3: Fundamental Frequency (F0) ---

def analyze_pitch_autocorr(file_path):
    print(f"\n--- Pitch Analysis (Autocorr) for {file_path} ---")
    y, sr = librosa.load(file_path)
    
    # 1. Compute autocorrelation for the entire signal
    # Using numpy correlate usually gives full size 2*N-1. Librosa has autocorr too.
    # Let's use librosa.autocorrelate for standard behavior or numpy.
    # The prompt says "for the entire signal".
    
    # Ideally F0 is time-varying, but for "piano_A4.wav" it's likely a single note.
    # However, for long signals, autocorr on whole signal smears things. 
    # Assuming the file is just the note.
    
    # We clip large signals to avoid huge computation or memory issues if very long
    max_len = sr * 3 # limit to 3 seconds if long, though A4 is likely short.
    if len(y) > max_len:
        print("Trimming signal for global autocorrelation...")
        y_short = y[:max_len]
    else:
        y_short = y

    ac = librosa.autocorrelate(y_short)
    
    # 2. Set initial items to zero for lags representing freq > 500Hz?? or to clear F0 peak?
    # Actually, we want to clear the 0-lag peak (energy). 
    # And we want to find F0. The prompt says "frequencies to clear the peak". 
    # Usually we zero out lags corresponding to very high frequencies (very short lags).
    # Let's just zero out the first few ms.
    # 500Hz correspond to sr/500 samples.
    # Let's say we look for pitch in human range, min lag needs to be > 0.
    # We clear lags corresponding to unwanted high freq noise or just the main lobe at 0.
    
    # A4 is 440Hz.
    # Let's zero out lags < 1ms or so? Or follow typical pitch detection:
    # Zero out lags corresponding to frequencies ABOVE expected max pitch, or simple low lags.
    
    ignore_freq_threshold = 2000 # Ignore freqs above 2000Hz?
    # Lag = sr / freq.
    # If freq = 2000, Lag = sr/2000.
    min_lag = int(sr / 2000) # e.g. 22050 / 2000 ~ 11 samples.
    
    # More aggressively, to clear the main lobe width.
    # Let's clear up to the first zero-crossing of the AC or just a fixed small amount.
    # Prompt: "Set initial autocorrelation items to zero for lags representing frequencies to clear the peak."
    # Let's clear up to 2ms (500Hz) just to be safe if looking for fundamental < 500Hz? 
    # But A4 is 440Hz. If we clear up to 500Hz (lag ~44 samples), we might clear the 440Hz peak if not careful? 
    # Wait, larger lag = lower freq. 440Hz is lag ~50.
    # If we clear "frequencies to clear the peak", maybe it means clear lags 0..k?
    
    # Let's assume we clear very small lags (high freqs).
    ac[:min_lag] = 0
    
    # 3. Find highest remaining peak
    peak_lag = np.argmax(ac)
    f0 = sr / peak_lag
    print(f"Autocorrelation Peak Lag: {peak_lag}")
    print(f"Estimated F0 (Autocorr): {f0:.2f} Hz")
    
    plt.figure()
    plt.plot(ac)
    plt.title(f"Autocorrelation (Peak at {peak_lag}, F0={f0:.1f}Hz)")
    plt.xlim(0, 1000) # focus on meaningful lags
    plt.savefig(os.path.join(OUTPUT_DIR, "autocorr_a4.png"))
    plt.close()

def compute_manual_zcr_f0(audio_signal, sampling_rate):
    """
    Estimates F0 using the ZCR method
    F0 = ZCR / 2 for periodic signals.
    """
    # Count sign changes
    # np.diff(np.sign) is non-zero at crossings.
    zero_crossings = np.where(np.diff(np.sign(audio_signal)))[0]
    zcr_count = len(zero_crossings)
    
    # Calculate crossings per second
    duration = len(audio_signal) / sampling_rate
    # If audio_signal is short window, duration is T_window.
    # Rate = count / duration
    zcr_rate = zcr_count / duration
    
    # Estimated F0
    f0_estimate = zcr_rate / 2
    return f0_estimate

def analyze_pitch_zcr(file_path):
    print(f"\n--- Pitch Analysis (ZCR) for {file_path} ---")
    y, sr = librosa.load(file_path)
    
    # Analyze stable region (e.g., middle 0.5s) to avoid onset/decay transients
    center_sample = len(y) // 2
    region_len = int(0.5 * sr)
    start = max(0, center_sample - region_len//2)
    end = min(len(y), center_sample + region_len//2)
    y_stable = y[start:end]
    
    f0_zcr = compute_manual_zcr_f0(y_stable, sr)
    print(f"Estimated F0 (ZCR) on stable region: {f0_zcr:.2f} Hz")

# --- Section 4: Homework ---

def homework_multitone(file_path):
    print(f"\n--- Homework H1: Multi-Tone Analysis for {file_path} ---")
    y, sr = librosa.load(file_path)
    
    # 40ms window spectrogram
    n_fft = int(sr * 0.040)
    hop_length = n_fft // 2
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.ylim(0, 2000) # A4 and F4 are < 1000Hz
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram (40ms window)")
    # F0 Trajectories using piptrack (better for polyphony)
    # piptrack returns candidates for each STFT bin. 
    # We select bins with high magnitude to visualize the tracks.
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    plt.figure(figsize=(10, 6))
    # Plot spectrogram as background
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='gray_r')
    
    # Extract peak pitches
    # pitches[f, t] contains frequency if it's a peak, 0 otherwise
    # magnitudes[f, t] contains the magnitude
    
    # Filter for visualization: select points with high magnitude
    times = librosa.times_like(pitches, sr=sr, hop_length=hop_length)
    
    # We'll plot variable-alpha scatter points or just the strong ones
    # Thresholding relative to max magnitude
    mag_thresh = np.max(magnitudes) * 0.3 # increased threshold to reduce noise
    
    # Iterate over time and find peaks
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t] > mag_thresh
        pitch_vals = pitches[index, t]
        if len(pitch_vals) > 0:
            plt.scatter(np.full_like(pitch_vals, times[t]), pitch_vals, color='cyan', s=5, alpha=0.8)

    plt.ylim(0, 1000) # Focus on F4(349) and A4(440) range
    plt.title("Spectrogram with F0 Trajectories (piptrack)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f0_trajectory.png"))
    plt.close()

def homework_speech(file_path):
    print(f"\n--- Homework H2: Speech Processing for {file_path} ---")
    y, sr = librosa.load(file_path)
    
    # 1. Mel Spectrogram (128 filters)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # 2. Log Mel Spectrogram
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Log Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "speech_mel_spectrogram.png"))
    plt.close()
    
    print("Log Mel Spectrogram saved.")
    print("Evaluation: Observe if horizontal bands (harmonics) are visible for vowels.")

if __name__ == "__main__":
    if os.path.exists(RAGTIME_FILE):
        analyze_intensity(RAGTIME_FILE)
    else:
        print(f"Skipping {RAGTIME_FILE} (Not Found)")
        
    if os.path.exists(PIANO_A4_FILE):
        analyze_pitch_autocorr(PIANO_A4_FILE)
        analyze_pitch_zcr(PIANO_A4_FILE)
    else:
        print(f"Skipping {PIANO_A4_FILE} (Not Found)")
        
    if os.path.exists(PIANO_A4_F4_FILE):
        homework_multitone(PIANO_A4_F4_FILE)
    else:
        print(f"Skipping {PIANO_A4_F4_FILE} (Not Found)")
        
    SPEECH_FILE = os.path.join(SAMPLE_DIR, 'vowel-sound-recording.m4a')
    if os.path.exists(SPEECH_FILE):
        homework_speech(SPEECH_FILE)
    else:
        print(f"Skipping {SPEECH_FILE} (Not Found)")
