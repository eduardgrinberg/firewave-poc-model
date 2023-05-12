import os
from pathlib import Path

import torchaudio

from audioUtil import AudioUtil

src_fold = '..\\esp32_raw_rate44_bit32'
out_fold = '..\\ready_esp32_44100_32-16'
target_len = 4

src_files = os.listdir(src_fold)


def save_segment(src, index, len, src_name, sample_rate):
    dst_name = Path(out_fold) / (Path(src_name).stem + '_' + index.__str__() + '.wav')
    torchaudio.save(dst_name, src[:, index:(index + len)], sample_rate, encoding='PCM_S', bits_per_sample=16)

    return dst_name


for file_name in src_files:
    print(f'Source file: {torchaudio.info(Path(src_fold) / file_name)}')
    samples, sr = torchaudio.load(Path(src_fold) / file_name)
    # samples, newsr = AudioUtil.resample(audio, target_sr)
    total_samples = samples.shape[1]
    segment_len = target_len * sr  # 4 seconds
    index = 0
    while total_samples - index > segment_len:
        out_file = save_segment(samples, index, segment_len, file_name, sr)
        index += segment_len

    print(f'Out file: {torchaudio.info(Path(out_file))}')



