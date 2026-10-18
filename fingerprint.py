"""
fingerprint.py
---
Audio fingerprinting library with optimised pathway, for a scaleable solution
Utilizes, byte packing hashes (cache efficiency), vectorised binary search (O(log N) search)
compressed, scaled down integer types for storage (to reduce memory footprint and faster access).
"""

import os #Standard Python library for scanning directories and handling file paths.
import numpy as np
import librosa #Audio processing library used here for loading audio and computing Short-Time Fourier Transforms (STFT)
from scipy.ndimage import maximum_filter

SR = 8000 #Target audio frequency is set to 8,000 samples per second
SONG_FOLDER = "songs"
DB_OUTPUT_PATH = "song_database.npz" #using numpy native format to utitlize the superfast vector operations. 

#loads the audio through librosa
def load_audio(path, target_sr=8000):
    try:
        data, sr = librosa.load(path, sr=target_sr, mono=True)
        max_val = np.max(np.abs(data)) #Computes absolute peak amplitude across all samples.
        if max_val > 0:
            data = data / (max_val + 1e-9)
        return data, sr
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")

#gets the list of songs
def list_songs(folder):
    if not os.path.isdir(folder):
        return []
    return [(os.path.splitext(f)[0], os.path.join(folder, f)) 
            for f in sorted(os.listdir(folder)) if f.lower().endswith(".mp3")]

#spectrogram generation using librosa stft, as it is faster in cpu than scipy.signal.stft
def spectrogram(audio, sr, nperseg=1024, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    hop_length = nperseg - noverlap
    stft_matrix = librosa.stft(audio, n_fft=nperseg, hop_length=hop_length, window='hann')
    Sxx = np.abs(stft_matrix)
    f = librosa.fft_frequencies(sr=sr, n_fft=nperseg)
    t = librosa.frames_to_time(np.arange(Sxx.shape[1]), sr=sr, hop_length=hop_length)
    return f, t, Sxx

#filtering the peak values to get the fingerprint
def find_peaks_2d(Sxx, amp_min_db=-25, neighborhood=(20, 5)):
    Sxx_db = 20 * np.log10(Sxx + 1e-9)
    Sxx_db -= Sxx_db.max()
    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    above_thresh = Sxx_db > amp_min_db
    freq_idx, time_idx = np.where(local_max & above_thresh)
    return np.column_stack((freq_idx, time_idx)).astype(np.int32) if len(freq_idx) > 0 else np.empty((0, 2), dtype=np.int32)

#optimising the hash size, by byte packing into int64 (a bit of overkill tbh)
def pack_hash(f1, f2, dt):
    return (np.uint64(f1) << 32) | (np.uint64(f2) << 16) | np.uint64(dt)

#hash generation function using peak values.
def generate_hashes(peaks, fan_out=5, min_dt=1, max_dt=50):
    if len(peaks) == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32)
    
    peaks = peaks[np.argsort(peaks[:, 1])]  
    hashes = []
    anchor_times = []
    
    for i, (f1, t1) in enumerate(peaks):
        count = 0
        for f2, t2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(pack_hash(f1, f2, dt))
            anchor_times.append(t1)
            count += 1
            if count >= fan_out:
                break
                
    return np.array(hashes, dtype=np.uint64), np.array(anchor_times, dtype=np.int32)

#ya, this is the part I match query with database, the intial python loop was too slow, so switch to searchsort of numpy
def match_query(audio, sr, db_hashes, db_songs, db_anchors, song_names, nperseg=1024, fan_out=5):
    _, _, Sxx = spectrogram(audio, sr, nperseg=nperseg)
    q_peaks = find_peaks_2d(Sxx)
    q_hashes, q_times = generate_hashes(q_peaks, fan_out=fan_out)

    if len(q_hashes) == 0 or len(db_hashes) == 0:
        return None, {}, q_peaks, 0, 0

    q_sort_idx = np.argsort(q_hashes)
    q_hashes = q_hashes[q_sort_idx]
    q_times = q_times[q_sort_idx]

    # binary search bounds lookup via np.searchsorted to skip O(N) masks and have O(log(N)) lookup
    left_bounds = np.searchsorted(db_hashes, q_hashes, side="left")
    right_bounds = np.searchsorted(db_hashes, q_hashes, side="right")

    valid_mask = right_bounds > left_bounds
    if not np.any(valid_mask):
        return None, {}, q_peaks, 0, 0
    
    
    l_intervals = left_bounds[valid_mask]
    r_intervals = right_bounds[valid_mask]
    matched_q_times_base = q_times[valid_mask]

    segment_lengths = r_intervals - l_intervals
    total_matches = segment_lengths.sum()

    db_indices = np.repeat(l_intervals, segment_lengths) + (
        np.arange(total_matches) - np.repeat(np.cumsum(segment_lengths) - segment_lengths, segment_lengths)
    )

    matched_db_songs = db_songs[db_indices]
    matched_db_anchors = db_anchors[db_indices]
    matched_q_times = np.repeat(matched_q_times_base, segment_lengths)

    offsets = matched_db_anchors - matched_q_times
    combined_votes = (offsets.astype(np.int64) << 32) | matched_db_songs.astype(np.int64)
    unique_keys, counts = np.unique(combined_votes, return_counts=True)
    
    song_scores = {}
    best_song_name = None
    best_score = 0
    best_offset = 0

    for key, count in zip(unique_keys, counts):
        s_idx = int(key & 0xFFFFFFFF)
        offset_val = int(key >> 32)
        s_name = song_names[s_idx]
        
        song_scores[s_name] = max(song_scores.get(s_name, 0), count)
        if count > best_score:
            best_score = count
            best_song_name = s_name
            best_offset = offset_val

    q_duration_bins = np.max(q_peaks[:, 1]) if len(q_peaks) > 0 else 0
    return best_song_name, song_scores, q_peaks, best_offset, q_duration_bins

#directly build the database from the songs, build once use multiple time,this custom databases is more suitable than redis and sqlite for this particular purpose.
if __name__ == "__main__":
    print(f"Compiling structural signatures from folder: '{SONG_FOLDER}' at {SR}Hz...")
    song_files = list_songs(SONG_FOLDER)
    if not song_files:
        print(f"Error: Add .mp3 assets to '{SONG_FOLDER}' before building.")
        exit(1)
        
    song_names = []
    collected_hashes = []
    collected_songs = []
    collected_anchors = []
    save_payload = {}

    for idx, (name, path) in enumerate(song_files):
        print(f" -> Processing structural constellations: {name}.mp3")
        song_names.append(name)
        
        audio, sr = load_audio(path, target_sr=SR)
        _, _, Sxx = spectrogram(audio, sr)
        peaks = find_peaks_2d(Sxx)
        
        save_payload[f"peaks_{idx}"] = peaks
        hashes, anchors = generate_hashes(peaks)
        
        if len(hashes) > 0:
            collected_hashes.append(hashes)
            # Optimization: Cast to np.uint8 since library has <= 256 songs
            collected_songs.append(np.full(len(hashes), idx, dtype=np.uint8))
            collected_anchors.append(anchors.astype(np.int32))

    if collected_hashes:
        db_hashes = np.concatenate(collected_hashes)
        db_songs = np.concatenate(collected_songs)
        db_anchors = np.concatenate(collected_anchors)
        
        sort_idx = np.argsort(db_hashes)
        save_payload["db_hashes"] = db_hashes[sort_idx]
        save_payload["db_songs"] = db_songs[sort_idx]
        save_payload["db_anchors"] = db_anchors[sort_idx]
        
        # Optimization: Precompute global metadata to eliminate run-time array scans
        song_hash_counts = np.bincount(db_songs, minlength=len(song_names))
        save_payload["song_hash_counts"] = song_hash_counts.astype(np.int32)
    else:
        save_payload["db_hashes"] = np.empty(0, dtype=np.uint64)
        save_payload["db_songs"] = np.empty(0, dtype=np.uint8)
        save_payload["db_anchors"] = np.empty(0, dtype=np.int32)
        save_payload["song_hash_counts"] = np.zeros(len(song_names), dtype=np.int32)

    save_payload["song_names"] = np.array(song_names, dtype=object)
    np.savez(DB_OUTPUT_PATH, **save_payload)
    print(f"\nSaved optimized database with {len(save_payload['db_hashes'])} hashes to '{DB_OUTPUT_PATH}'")
