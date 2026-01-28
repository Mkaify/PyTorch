import librosa
import numpy as np
import os
from scipy.ndimage import maximum_filter
from collections import defaultdict

# Parameters
FAN_VALUE = 15  # how many neighboring peaks to pair
SAMPLE_RATE = 22050

# Step 1: Generate fingerprints from a spectrogram
def generate_fingerprints(y, sr):
    S = np.abs(librosa.stft(y))
    log_s = librosa.amplitude_to_db(S, ref=np.max)
    peaks = get_peaks(log_s)
    fingerprints = []

    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE):
            if i + j < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                delta_t = time2 - time1
                if 0 < delta_t <= 10:
                    h = (freq1, freq2, delta_t)
                    fingerprints.append((h, time1))
    return fingerprints

# Step 2: Peak detection
def get_peaks(spectrogram, threshold=10):
    struct = np.ones((3, 3))
    local_max = maximum_filter(spectrogram, footprint=struct) == spectrogram
    detected = np.where((spectrogram > threshold) & local_max)
    return list(zip(detected[0], detected[1]))

# Step 3: Build fingerprint database
db = defaultdict(list)

def index_song(path, song_id):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    fingerprints = generate_fingerprints(y, sr)
    for fp, t in fingerprints:
        db[fp].append((song_id, t))

# Step 4: Query matching
def recognize_snippet(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    fingerprints = generate_fingerprints(y, sr)
    matches = defaultdict(int)
    for fp, t in fingerprints:
        if fp in db:
            for song_id, t_db in db[fp]:
                offset = t_db - t
                matches[(song_id, offset)] += 1

    if not matches:
        return "No match found"
    best = max(matches, key=matches.get)
    return f"Match: {best[0]} (offset {best[1]}) — Score: {matches[best]}"

# Index known songs
index_song("data/songs/song1.wav", "Song 1")
index_song("data/songs/song2.wav", "Song 2")

# Recognize short snippet
result = recognize_snippet("data/snippets/query.wav")
print("🎵 Result:", result)