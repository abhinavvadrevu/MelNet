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

def process_blizzard(text: str):
    original_text = text
    text = text.replace('@ ', '').replace('# ', '').replace('| ', '')
    # THE FOLLOWING LINE ONLY EXISTS FOR CLEANED BLIZZARD
    # BECAUSE I'M TRYING TO REUSE WEIGHTS FROM EXISTIN DATA WHICH DIDN'T
    # HAVE QUOTES. EVENTUALLY, I DO NEED TO FIND A WAY TO SUPPORT THIS
    # SYMBOL AS IT'S A CRITICAL ONE IN DEFINING HOW SPEECH SOUNDS
    text = text.replace('"', '')
    text = text.replace(']', '')
    text = text.replace('[', '')
    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent ')
    text = text.replace('$', ' dollars ')


    no_punc_text = text.replace(',','').replace('.','')
    easy_numbers = [int(i) for i in no_punc_text.split() if i.isdigit()]
    # Now find and replace ordinal numbers
    ordinal_number_strs = [
        i for i in no_punc_text.split() if
        (i.endswith('st') or i.endswith('th') or i.endswith('rd') or i.endswith('nd')) and
        (i.replace('st','').replace('th','').replace('rd','').replace('nd','').isdigit())
    ]

    easy_numbers = sorted(easy_numbers, key=lambda x: -1*len(str(x)))
    ordinal_number_strs = sorted(ordinal_number_strs, key=lambda x: -1*len(str(x)))

    for number in easy_numbers:
        number_text = inflect_engine.number_to_words(number)
        if number > 1000 and number < 2200:
            number_text = inflect_engine.number_to_words(number, group=2)
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(str(number), number_text)

    for ordinal_number in ordinal_number_strs:
        number = int(ordinal_number.replace('st','').replace('th','').replace('rd','').replace('nd',''))
        number_text = num2words(number, to='ordinal')
        number_text = number_text.replace(',', '')
        number_text = ' ' + number_text.replace('-', ' ') + ' '
        text = text.replace(ordinal_number, str(number))
        text = text.replace(str(number), number_text)

    text = text.replace('  ', ' ')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' !', '!')
    text = text + EOS
    
    seq = None
    try:
        seq = [_symbol_to_id[c] for c in text]
    except Exception as e:
        print(original_text)
        print(text)
        return False
    return True

txt_train_path = os.path.join('datasets', 'cleaned_blizzard/train_txt')
txt_test_path = os.path.join('datasets', 'cleaned_blizzard/test_txt')
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
    success = process_blizzard(sentence)