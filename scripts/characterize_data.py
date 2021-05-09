import glob
import librosa
import os
import plotille

# Define path
wav_path = 'datasets/train_wav'
txt_path = 'datasets/train_txt'
old_root_dir = './datasets'

# Num samples

def get_sentences(root_dir):
  dataset = []
  with open(os.path.join(root_dir, 'prompts.gui'), 'r') as f:
    lines = f.read().splitlines()
    filenames = lines[::3]
    sentences = lines[1::3]
    for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
        wav_path = os.path.join(root_dir, 'wavn', filename + '.wav')
        length = get_length(wav_path, hp.audio.sr)
        if length < hp.audio.duration:
            dataset.append((wav_path, sentence))
  return dataset

wav_file_list = glob.glob(
    os.path.join(wav_path, '*.wav'),
    recursive=True
)
txt_file_list = glob.glob(
    os.path.join(txt_path, '*.txt'),
    recursive=True
)
old_dataset = get_sentences(old_root_dir)
print("Number of wav samples in %s: %d" % (wav_path, len(wav_file_list)))
print("Number of txt samples in %s: %d" % (txt_path, len(txt_file_list)))
print("Number of txt/wav samples in original code: %d" % len(old_dataset))

# Wav legth histogram
wav_lengths = map(lambda file: librosa.get_duration(filename=file), filelist)
plotille.hist(wav_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')

old_wav_lengths = map(lambda data: librosa.get_duration(filename=data[0]), old_dataset)
plotille.hist(old_wav_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')

# Max wav length
print("Max wav length: %f" % max(wav_lengths))
print("Max old wav length: %f" % max(old_wav_lengths))

# Min wav length
print("Min wav length: %f" % min(wav_lengths))
print("Min old wav length: %f" % min(old_wav_lengths))

# Text character lengths histogram
def read_txt_file(filepath):
  with open(filepath, 'r') as reader:
    sentence = reader.read().strip()
    return sentence

sentences = map(read_txt_file, filelist)
sentence_lengths = map(len, sentences)
old_sentences = map(lambda x:x[1], old_dataset)
old_sentence_lengths = map(len, old_sentences)

plotille.hist(sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
plotille.hist(old_sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')

# Max text length
print("Max txt length: %f" % max(sentence_lengths))
print("Max old txt length: %f" % max(old_sentence_lengths))

# Min text length
print("Min txt length: %f" % min(sentence_lengths))
print("Min old txt length: %f" % min(old_sentence_lengths))


