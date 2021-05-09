import glob
import librosa
import os
import plotille
from tqdm import tqdm
import audiosegment

# Define path
wav_path = 'datasets/complete_blizzard/train_wav'
txt_path = 'datasets/complete_blizzard/train_txt'
old_root_dir = './datasets/segmented'

# Num samples
def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

# def get_sentences(root_dir):
#   dataset = []
#   with open(os.path.join(root_dir, 'prompts.gui'), 'r') as f:
#     lines = f.read().splitlines()
#     filenames = lines[::3]
#     sentences = lines[1::3]
#     for filename, sentence in tqdm(list(zip(filenames, sentences)), total=len(filenames)):
#         wav_path = os.path.join(root_dir, 'wavn', filename + '.wav')
#         length = get_length(wav_path, 7000)
#         if length < 6:
#             dataset.append((wav_path, sentence))
#   return dataset

def read_txt_file(filepath):
  with open(filepath, 'r') as reader:
    sentence = reader.read().strip()
    return sentence

wav_file_list = glob.glob(
    os.path.join(wav_path, '*.wav'),
    recursive=True
)
txt_file_list = glob.glob(
    os.path.join(txt_path, '*.txt'),
    recursive=True
)

new_dataset = []
new_wav_lengths = []
new_sentences = []
for wav_file in tqdm(wav_file_list, total=len(wav_file_list)):
  txt_file = wav_file.replace('wav', 'txt')
  wav_length = get_length(wav_file, 7000)
  if wav_length < 6:
    sentence = read_txt_file(txt_file)
    new_dataset.append((wav_file, sentence))
    new_wav_lengths.append(wav_length)
    new_sentences.append(sentence)

# Write to file
with open('datasets/complete_blizzard/train_prompts.gui', 'w') as writer:
  for wav_file, sentence in new_dataset:
    writer.write(wav_file)
    writer.write('\n')
    writer.write(sentence)
    writer.write('\n')