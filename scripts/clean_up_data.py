import editdistance
import glob
import boto3
import random
import os
import shutil
from tqdm import tqdm
import audiosegment
# Existing directory structure:
# - datasets
#   - segmented_training_data
#   - BC2013_segmented_v0_txt1
#   - BC2013_segmented_v0_wav1
#   - BC2013_segmented_v0_txt2
#   - BC2013_segmented_v0_wav2
#   - BC2013_segmented_v1_txt_selection
#   - BC2013_segmented_v1_wav_selection

# New directory structure
# - datasets
#   - train_wav
#   - train_txt
#   - test_wav
#   - test_txt

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


def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

def parse_segmented():
  # parses segmented_training_data
  # splits into train and test based on existing split
  SAMPLE_RATE = 22050
  MAX_DURATION = 10.0 # Max duration 10 seconds for now
  dataset = []
  train_dataset = []
  test_dataset = []
  root_dir = 'datasets/segmented_training_data'
  with open(os.path.join(root_dir, 'prompts.gui'), 'r') as f:
    lines = f.read().splitlines()
    filenames = lines[::3]
    sentences = lines[1::3]
    for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
      wav_path = os.path.join(root_dir, 'wavn', filename + '.wav')
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

  return

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

list_of_sentences = []
def sentence_exists(new_sentence):
  global list_of_sentences
  # checks if a given sentence already exists in train/test set
  file_list = glob.glob(
      os.path.join('datasets/train_txt','*'),
      recursive=True
  ) + glob.glob(
      os.path.join('datasets/test_txt','*'),
      recursive=True
  )
  if len(file_list) == 0:
    return False
  if len(list_of_sentences) == 0: # TODO: Update this memoized list as we add new sentences in
    sentences = []
    for filepath in file_list:
      sentence = read_txt_file(filepath)
      sentences.append(sentence)
    list_of_sentences = sentences

  for sentence in list_of_sentences:
    if new_sentence.replace(" ", "").lower() == sentence.replace(" ", "").lower():
      print("Sentence exists! Tried adding \"%s\" to dataset with \"%s\""% (new_sentence, sentence))
      return True
  return False

def parse_new_data():
  # reads new data
  # removes duplicates
  # saves data into the right places

  # Start with v0_txt1
  txt_file_list = glob.glob(
      os.path.join('datasets/BC2013_segmented_v1_txt_selection','**', '*.txt'),
      recursive=True
  )
  wav_file_list = glob.glob(
      os.path.join('datasets/BC2013_segmented_v1_wav_selection','**', '*.wav'),
      recursive=True
  )

  data_to_save = []
  i = 0
  for txt_file in txt_file_list:
    i += 1
    # first check if sentence is already in the dataset
    print("Processing file %d: %s" % (i, txt_file))
    sentence = read_txt_file(txt_file)
    if sentence_exists(sentence):
      print("%s already exists!" % sentence)
      continue
    txt_filename = os.path.basename(txt_file)
    wav_filename = os.path.basename(txt_file).replace('.txt', '.wav')
    for wav_file in wav_file_list:
      if wav_file.endswith(wav_filename):
        # print("Found matching wav file! %s" % wav_filename)
        # data_to_save.append((txt_file, wav_file))
        if 'testset' in txt_file:
          save_new_file(wav_file, read_txt_file(txt_file), False)
        else:
          save_new_file(wav_file, read_txt_file(txt_file), True)

def compress():
  # creates compressed version of data at 15000 SR to help with memory
  pass

def upload_file(local_file_path, s3_file_path, bucket):
    s3_client = boto3.client(
        's3'
    )
    try:
        response = s3_client.upload_file(local_file_path, bucket, s3_file_path)
    except ClientError as e:
        print("ERROR WITH UPLOAD")
        print(e)
        return False
    return True


# parse_new_data()

def upload_dataset():
  upload_file('datasets/cleaned_blizzard.zip', 'cleaned_blizzard.zip', 'blizzard2013')

upload_dataset()