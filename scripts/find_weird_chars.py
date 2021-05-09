import os
import glob
import random
import numpy as np
import subprocess
import audiosegment
import inflect
from num2words import num2words

inflect_engine = inflect.engine()
PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?`'
SPACE = ' '
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
en_symbols = SYMBOLS + PAD + EOS + PUNC + SPACE
_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}

def remove_punc(text):
    return text \
    .replace(',','') \
    .replace('.','') \
    .replace('?','') \
    .replace('*','') \
    .replace(')','') \
    .replace('(','') \
    .replace('[','') \
    .replace(']','') \
    .replace('/','') \
    .replace('{','') \
    .replace('}','') \
    .replace(';','') \
    .replace(':','') \
    .replace('&','')

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

        # print(txt_filepath)
        # print(original_text)
        # print(text)
        # print('')

    for ordinal_number, ordinal_number_str in ordinal_number_strs:
        number_text = num2words(ordinal_number, to='ordinal')
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(ordinal_number_str, number_text)

        # print(txt_filepath)
        # print(original_text)
        # print(text)
        # print('')

    for degree_number in degrees_number_strs:
        number = int(degree_number.replace('deg','').replace('d',''))
        number_text = inflect_engine.number_to_words(number)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(str(degree_number), number_text + ' degrees')

        # print(txt_filepath)
        # print(original_text)
        # print(text)
        # print('')

    for pound_number, pount_str in pounds_number_strs:
        number_text = inflect_engine.number_to_words(pound_number)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(pount_str, number_text + ' pounds')

        # print(txt_filepath)
        # print(original_text)
        # print(text)
        # print('')

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

txt_train_path = os.path.join('datasets', 'complete_blizzard/train_txt')
txt_test_path = os.path.join('datasets', 'complete_blizzard/test_txt')
txt_file_list = glob.glob(
    os.path.join(txt_train_path, '**', '*.txt'),
    recursive=True
) + glob.glob(
    os.path.join(txt_test_path, '**', '*.txt'),
    recursive=True
)
for txt_filepath in txt_file_list:
    f = open(txt_filepath, "r")
    sentence = f.read().strip()
    f.close()
    success = process_blizzard(sentence, txt_filepath)