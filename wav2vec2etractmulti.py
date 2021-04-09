from pathlib import Path

import soundfile as sf
import numpy as np
import torch as th
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from torchaudio.transforms import Resample
from tqdm import tqdm

from ray.util import ActorPool

class Wav2VecTokenizer:
    def __init__(self):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def tokenize(self, audio_path):
        audio_input, sampling_rate = sf.read(audio_path)
        resampler = Resample(sampling_rate)
        audio_input = resampler(th.tensor(audio_input))
        input_values = self.tokenizer(audio_input, return_tensors="pt").input_values
        return input_values


class Wav2VecExtract:
    def __init__(self, device):
        self.device = device

        self.raw_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)

    def predict(self, input_values, features_path):
        input_values = input_values.to(self.device)
        hidden_state = self.raw_model(input_values).last_hidden_state
        hidden_state = hidden_state.flatten()
        hidden_state = hidden_state.cpu().detach().numpy()
        np.save(features_path, hidden_state)


if __name__ == "__main__":
    tokenizer = Wav2VecTokenizer()
    extractor = Wav2VecExtract(th.device("cuda"))
    audio_dir = Path("/impressionv2_faces/audio/")
    videos = list(audio_dir.glob("*.wav"))

    input_values_list = []
    for video_path in tqdm(videos):
        video_name = video_path.stem
        feature_path = audio_dir / f"{video_name}_wav2vec2.npy"
        input_values = tokenizer.tokenize(video_path)
        input_values_list.append(input_values)

    for input_values in input_values_list:
        extractor.predict(video_path, feature_path)
