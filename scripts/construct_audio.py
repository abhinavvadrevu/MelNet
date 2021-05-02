import os
import glob
import argparse
import torch
import audiosegment
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import plot_spectrogram_to_numpy
from utils.reconstruct import Reconstruct
from utils.constant import t_div
from utils.hparams import HParam
from model.model import MelNet
import librosa
import soundfile as sf




generated = torch.load('../Melnet files/a100_blizzard_compressed/take_3/hw_blizzard_compressed.pt', map_location=torch.device('cpu'))
# generated = torch.load('../../hw_blizzard_compressed.pt')

generated_np = generated[0].cpu().detach().numpy()



# turn inference ms back to audio
def denormalize(x):
  return (np.clip(x, 0.0, 1.0) - 1.0) * 80.0

x = librosa.db_to_power(denormalize(generated_np) + 20.0)
y = librosa.feature.inverse.mel_to_audio(
  M=x,
  sr=7000,
  n_fft=1080,
  hop_length=150,
  win_length=1080
)
sf.write('../Melnet files/librosa_testing/hello_world_blizzard_reconstructed2.wav', y, 7000, 'PCM_24')




y, sr = librosa.load('../Melnet files/librosa_testing/hello_world.mp3')
ms = librosa.feature.melspectrogram(y, sr)
spectrogram = plot_spectrogram_to_numpy()


# turn my voice's ms back to audio
S = librosa.feature.inverse.mel_to_stft(ms)
y = librosa.griffinlim(S)
sf.write('../Melnet files/librosa_testing/hello_world_reconstructed.wav', y, sr, 'PCM_24')
