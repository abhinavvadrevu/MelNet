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

def get_sentences(root_dir):
  dataset = []
  with open(os.path.join(root_dir, 'prompts.gui'), 'r') as f:
    lines = f.read().splitlines()
    filenames = lines[::3]
    sentences = lines[1::3]
    for filename, sentence in tqdm(list(zip(filenames, sentences)), total=len(filenames)):
        wav_path = os.path.join(root_dir, 'wavn', filename + '.wav')
        length = get_length(wav_path, 7000)
        if length < 6:
            dataset.append((wav_path, sentence))
  return dataset

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

old_dataset = get_sentences(old_root_dir)
print("Number of txt/wav samples in %s: %d" % (wav_path, len(new_dataset)))
print("Number of txt/wav samples in original code: %d" % len(old_dataset))

# Wav legth histogram
hist1 = plotille.hist(new_wav_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print("New wav lengths histogram")
print(hist1)
print('')

old_wav_lengths = list(map(lambda data: librosa.get_duration(filename=data[0]), old_dataset))
hist2 = plotille.hist(old_wav_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print("New wav lengths histogram")
print(hist2)
print('')

# Max wav length
print("Max wav length: %f" % max(new_wav_lengths))
print("Max old wav length: %f" % max(old_wav_lengths))

# Min wav length
print("Min wav length: %f" % min(new_wav_lengths))
print("Min old wav length: %f" % min(old_wav_lengths))

# Total wav length
print("Total wav length (seconds): %f" % sum(new_wav_lengths))
print("Total wav length (hours): %f" % (sum(new_wav_lengths)/60.0/60.0))
print("Total old wav length (seconds): %f" % sum(old_wav_lengths))
print("Total old wav length (hours): %f" % (sum(old_wav_lengths)/60.0/60.0))

# Text character lengths histogram
sentence_lengths = list(map(len, new_sentences))
old_sentences = list(map(lambda x:x[1], old_dataset))
old_sentence_lengths = list(map(len, old_sentences))

hist1 = plotille.hist(sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print(hist1)
hist2 = plotille.hist(old_sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print(hist2)

# Max text length
print("Max txt length: %f" % max(sentence_lengths))
print("Max old txt length: %f" % max(old_sentence_lengths))

# Min text length
print("Min txt length: %f" % min(sentence_lengths))
print("Min old txt length: %f" % min(old_sentence_lengths))


