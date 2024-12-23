import numpy as np
import librosa

def extract_music_features(audio_path: str):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

    return(
        {
            'y_harmonic': y_harmonic,
            'y_percussive': y_percussive,
            'tempo': tempo,
            'beat_frames': beat_frames,
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'beat_mfcc_delta': beat_mfcc_delta,
            'chromagram': chromagram,
            'beat_chroma': beat_chroma,
            'beat_features': beat_features
        }
    )

# if __name__ == "__main__":
#     extract_music_features()
# ~/Music/Music/Media.localized/Music/Unknown\ Artist/Unknown\ Album/01\ S1\ 1.mp3

from pathlib import Path
path = Path('rawdata/01_S1_1.mp3')
print(path)
extract_music_features(path)