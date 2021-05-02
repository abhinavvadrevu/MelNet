# Read file
# sort it all out
import os
import glob
import shutil
from tqdm import tqdm
import audiosegment
import random


SAMPLE_RATE = 22050
MAX_DURATION = 10.0 # Max duration 10 seconds for now

def read_txt_file(filepath):
  f = open(filepath, "r")
  sentence = f.read().strip()
  f.close()
  return sentence

def write_txt_file(filepath, sentence):
  f = open(filepath, "w")
  f.write(sentence)
  f.close()
  return sentence

def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds


def get_last_index(train=False):
  path = 'datasets/train_txt'
  if not train:
    path = 'datasets/test_txt'
  file_list = glob.glob(
      os.path.join(path,'**', '*.txt'),
      recursive=True
  )
  last_index = 0
  # Assumes we always have 5 digits in naming
  if len(file_list) > 0:
    last_index = max(map(lambda x: int(x[-9:-4]), file_list))
  return last_index

def save_new_file(wavpath, sentence, train=False):
  # saves a new file into our new dataset
  # First get last index number
  last_index = get_last_index(train)
  next_index = last_index + 1
  new_index_str = f'{next_index:05}'
  train_str = 'train' if train else 'test'
  txt_filepath = 'datasets/%s_txt/%s.txt' % (train_str, new_index_str)
  wav_filepath = 'datasets/%s_wav/%s.wav' % (train_str, new_index_str)
  # First save the txt in the right place
  write_txt_file(txt_filepath, sentence)
  # now copy the wav file into the right place
  old_wav_file_name = os.path.basename(wavpath)
  new_wav_file_path = os.path.dirname(wav_filepath)
  shutil.copy(wavpath, new_wav_file_path)
  os.rename(os.path.join(new_wav_file_path, old_wav_file_name), wav_filepath)

dataset = []
with open(os.path.join('datasets/segmented', 'prompts.gui'), 'r') as f:
  lines = f.read().splitlines()
  filenames = lines[::3]
  sentences = lines[1::3]
  for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
    wav_path = os.path.join('datasets/segmented', 'wavn', filename + '.wav')
    length = get_length(wav_path, SAMPLE_RATE)
    if length < MAX_DURATION:
      sentence = sentence.replace('@ ', '').replace('# ', '').replace('| ', '')
      dataset.append((wav_path, sentence))

random.seed(123)
random.shuffle(dataset)
train_dataset = dataset[:int(0.95 * len(dataset))]
test_dataset = dataset[int(0.95 * len(dataset)):]

for wavpath, sentence in train_dataset:
  save_new_file(wavpath, sentence, True)

for wavpath, sentence in test_dataset:
  save_new_file(wavpath, sentence, False)