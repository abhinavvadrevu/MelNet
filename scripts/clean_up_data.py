import editdistance
import glob
import boto3
import random
import os
import shutil
from tqdm import tqdm
import audiosegment
import json
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

def process_blizzard(text: str, txt_filepath):
    original_text = text
    text = text.replace('@ ', '').replace('# ', '').replace('| ', '')
    # THE FOLLOWING LINE ONLY EXISTS FOR CLEANED BLIZZARD
    # BECAUSE I'M TRYING TO REUSE WEIGHTS FROM EXISTIN DATA WHICH DIDN'T
    # HAVE QUOTES. EVENTUALLY, I DO NEED TO FIND A WAY TO SUPPORT THIS
    # SYMBOL AS IT'S A CRITICAL ONE IN DEFINING HOW SPEECH SOUNDS
    text = text.replace('"', '')
    text = text.replace(']', '')
    text = text.replace('[', '')
    text = text.replace('/', '')
    text = text.replace('}', '')
    text = text.replace('{', '')
    text = text.replace('*', '')
    text = text.replace('<c', '')
    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent ')
    text = text.replace('$', ' dollars ')


    no_punc_text = text.replace(',','').replace('.','').replace('?','')
    easy_numbers = [
        (int(remove_punc(i)), i)
        for i in text.split()
        if remove_punc(i).isdigit()
    ]
    # Now find and replace ordinal numbers
    ordinal_number_strs = [
        (int(remove_punc(i).replace('st','').replace('th','').replace('rd','').replace('nd','')), i)
        for i in text.split() if
        (
            remove_punc(i).endswith('st') or
            remove_punc(i).endswith('th') or
            remove_punc(i).endswith('rd') or
            remove_punc(i).endswith('nd')
        ) and
        (
            remove_punc(i) \
            .replace('st','') \
            .replace('th','') \
            .replace('rd','') \
            .replace('nd','') \
            .isdigit())
    ]
    degrees_number_strs = [
        i for i in no_punc_text.split() if
        (i.endswith('deg') or i.endswith('d')) and
        i.replace('deg','').replace('d','').isdigit()
    ]

    pounds_number_strs = [
        (int(remove_punc(i).replace('L','')), i)
        for i in text.split() if
        (i.endswith('L') or i.startswith('L')) and
        remove_punc(i).replace('L','').isdigit()
    ]

    easy_numbers = sorted(easy_numbers, key=lambda x: -1*len(str(x)))
    ordinal_number_strs = sorted(ordinal_number_strs, key=lambda x: -1*len(str(x)))
    degrees_number_strs = sorted(degrees_number_strs, key=lambda x: -1*len(str(x)))
    pounds_number_strs = sorted(pounds_number_strs, key=lambda x: -1*len(str(x)))

    for number, number_str in easy_numbers:
        number_text = inflect_engine.number_to_words(number)
        if number > 1000 and number < 2200:
            number_text = inflect_engine.number_to_words(number, group=2)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(number_str, number_text)

    for ordinal_number, ordinal_number_str in ordinal_number_strs:
        number_text = num2words(ordinal_number, to='ordinal')
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(ordinal_number_str, number_text)

    for degree_number in degrees_number_strs:
        number = int(degree_number.replace('deg','').replace('d',''))
        number_text = inflect_engine.number_to_words(number)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(str(degree_number), number_text + ' degrees')

    for pound_number, pount_str in pounds_number_strs:
        number_text = inflect_engine.number_to_words(pound_number)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(pount_str, number_text + ' pounds')

    text = text.replace('  ', ' ')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' !', '!')
    text = text + EOS
    
    seq = None
    try:
        seq = [_symbol_to_id[c] for c in text]
    except Exception as e:
        print(txt_filepath)
        print(original_text)
        print(text)
        print('')
        return False
    return True

def save_new_sentence(original_sentence, parsed_sentence, wav_path, wav_length, train=True):
  to_save = {
    'original_sentence': original_sentence,
    'parsed_sentence': parsed_sentence,
    'wav_path': wav_path,
    'wav_length': wav_length,
    'train': train,
  }
  with open('./datasets/blizzard.json', 'r+') as json_file:
    data = json.load(json_file)
    data.append(to_save)
    json_file.seek(0)
    json.dump(data, json_file)


def get_length(wavpath, sample_rate=22050):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

def read_txt_file(filepath):
  with open(filepath, 'r') as reader:
    sentence = reader.read().strip()
    return sentence

def write_txt_file(filepath, sentence):
  f = open(filepath, "w")
  f.write(sentence)
  f.close()
  return sentence

def parse_new_data(txt_path, wav_path):
  # reads new data
  # removes duplicates
  # saves data into the right places

  # Start with v0_txt1
  txt_file_list = glob.glob(
      os.path.join(txt_path,'**', '*.txt'),
      recursive=True
  )
  wav_file_list = glob.glob(
      os.path.join(wav_path,'**', '*.wav'),
      recursive=True
  )

  data_to_save = []
  i = 0
  for txt_file in tqdm(txt_file_list, desc="Processing data files"):
    i += 1
    # first check if sentence is already in the dataset
    # print("\n\nProcessing file %d: %s" % (i, txt_file))
    sentence = read_txt_file(txt_file)
    txt_filename = os.path.basename(txt_file)
    wav_filename = os.path.basename(txt_file).replace('.txt', '.wav')
    wav_file = txt_file.replace('txt', 'wav')
    wav_length = get_length(wav_file)
    processed_sentence = process_blizzard(sentence, txt_file)
    # Save txt file with length of audio AND processed sentence
    train = False if random.random() < 0.05 else True
    save_new_sentence(sentence, processed_sentence, wav_file, wav_length, train=True)

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


# parse_new_data('datasets/BC2013_segmented_v0_txt1', 'datasets/BC2013_segmented_v0_wav1')
# parse_new_data('datasets/BC2013_segmented_v0_txt2', 'datasets/BC2013_segmented_v0_wav2')

def upload_dataset():
  upload_file('datasets/complete_blizzard.zip', 'complete_blizzard.zip', 'blizzard2013')

upload_dataset()