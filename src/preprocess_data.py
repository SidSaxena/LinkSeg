import argparse
import madmom
import os
import numpy as np
import librosa
import shutil
from pathlib import Path
import multiprocessing as mp


from tqdm import tqdm
from pydub import AudioSegment
from data_utils import FileStruct, find_audio, write_beats


def wav_cache_path(file_struct):
    """Decoded-audio path, under the output directory rather than beside the input.

    Keeping it out of the audio directory stops a later run from finding both a
    track and its own conversion.
    """
    cache = Path(file_struct.out_path).joinpath('wav_cache')
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache.joinpath(file_struct.track_name + '.wav'))


def to_wav(file_struct):
    """Decode to wav in the cache, leaving the source file untouched.

    The source was previously deleted once converted, which destroyed the input
    audio and left a second run reading wav where the first had read mp3.
    """
    dst = wav_cache_path(file_struct)
    if not os.path.isfile(dst):
        AudioSegment.from_file(file_struct.audio_file).export(dst, format='wav')
    return dst


def madmom_beats(file_struct, y_, sr):

    if Path(file_struct.audio_file).suffix.lower() != '.wav':
        audiofile = to_wav(file_struct)
        file_struct.audio_file = audiofile
        y_, sr = librosa.load(audiofile, mono=True)

    sr_ = sr
    if sr != 44100:
        y_ = librosa.resample(y_, orig_sr=sr, target_sr=44100)
        sr_ = 44100

    audio_duration = librosa.get_duration(y=y_, sr=sr_)
    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(y_)
    beat_times = np.asarray(proc(act))
    if beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0)
    new_beats = []
    for i in range(len(beat_times)):
        if beat_times[i] < audio_duration:
            new_beats.append(beat_times[i])
    beat_times = new_beats
    beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beats, audio_duration


def compute_beats(file_struct, y, sr):
    beat_frames, duration = madmom_beats(file_struct, y, sr)
    write_beats(file_struct, beat_frames, duration)


def process_beats(file_struct):
    if not os.path.isfile(file_struct.beat_file):
        y, sr = librosa.load(file_struct.audio_file, mono=True)
        compute_beats(file_struct, y, sr)
    else:
        print('Beats already found, skipping.')
    return 0


def get_paths(ds_path, config):
    tracklist = find_audio(ds_path, ext=config.dataset.audio_exts)
    npy_path = os.path.join(ds_path, 'audio_npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    return tracklist, npy_path


def get_npy(fn):
    x, _ = librosa.core.load(fn, sr=22050)
    return x


def process_audio(file_struct):
    print(file_struct.audio_npy_file, os.path.exists(file_struct.audio_npy_file))
    if not os.path.exists(file_struct.audio_npy_file):
        x = get_npy(file_struct.audio_file)
        np.save(open(file_struct.audio_npy_file, 'wb'), x)


def wav_conversion(file, output_path=None):
    """Wav path for file, decoding through the cache when it is not already wav."""
    if Path(file).suffix.lower() == '.wav':
        return file
    return to_wav(FileStruct(file, output_path))



def process_track(track, output_path=None):
    file_struct = FileStruct(track, output_path)
    process_audio(file_struct)
    process_beats(file_struct)

def preprocess_data_(args):
    output_path = getattr(args, 'output_path', None)
    tracklist = find_audio(args.data_path)
    pool = mp.Pool(mp.cpu_count())
    funclist = []
    for file in tqdm(tracklist):
        f = pool.apply_async(process_track, [file, output_path])
        funclist.append(f)
    pool.close()
    pool.join()

def preprocess_data(args):
    # Passed to each worker explicitly rather than held in module state, because
    # a spawned pool re-imports this module and would not inherit it.
    output_path = getattr(args, 'output_path', None) or args.data_path
    tracklist = find_audio(args.data_path)
    pool = mp.Pool(mp.cpu_count())
    npy_path = os.path.join(output_path, 'audio_npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    funclist = []
    for file in tqdm(tracklist):
        f = pool.apply_async(process_track, [file, output_path])
        funclist.append(f)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str, default=None,
                        help='Directory for audio_npy/ and features/. '
                             'Defaults to --data_path.')
    args = parser.parse_args()
    print(args)
    preprocess_data(args)
