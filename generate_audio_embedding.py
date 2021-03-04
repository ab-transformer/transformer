import os
from pathlib import Path

import soundfile as sf
import numpy as np
import torch as th
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from torchaudio.transforms import Resample

IMPRESSIONV2_DIR = Path("/impressionv2")
EMBEDDING_DIR = Path("/mbalazsdb")


class AudioEmbedding:
    def __init__(self, device):
        self.device = device

        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.raw_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(self.device)

    def generate_embedding_of_file(self, file_name):
        audio_input, sampling_rate = sf.read(file_name)
        resampler = Resample(sampling_rate)
        audio_input = resampler(th.tensor(audio_input))
        input_values = self.tokenizer(audio_input, return_tensors="pt").input_values.to(
            self.device
        )
        hidden_state = self.raw_model(input_values).last_hidden_state
        hidden_state = hidden_state.flatten()
        return hidden_state.cpu().detach().numpy()

    def generate_embedding_for_split(self, split):
        split_impressionv2_dir = IMPRESSIONV2_DIR / split
        split_embedding_dir = EMBEDDING_DIR / "audio" / split
        os.makedirs(split_embedding_dir, exist_ok=True)

        for video_id in os.listdir(split_impressionv2_dir):
            embedding = self.generate_embedding_of_file(
                split_impressionv2_dir / video_id / f"{video_id}.wav"
            )
            np.save(split_embedding_dir / f"{video_id}.npy", embedding)


if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    embedding_model = AudioEmbedding(device)
    embedding_model.generate_embedding_for_split("train")
    embedding_model.generate_embedding_for_split("test")
    embedding_model.generate_embedding_for_split("valid")
