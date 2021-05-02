import random
import numpy as np
import subprocess
import audiosegment
import inflect

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?`'
SPACE = ' '
SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
en_symbols = SYMBOLS + PAD + EOS + PUNC + SPACE
_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}
inflect_engine = inflect.engine()

def get_length(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    return audio.duration_seconds

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
        raise e
    return np.array(seq, dtype=np.int32)

def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(wavpath, sample_rate):
    audio = audiosegment.from_file(wavpath).resample(sample_rate_Hz=sample_rate)
    wav = audio.to_numpy_array()
    
    if len(wav.shape) == 2:
        wav = wav.T.flatten()
    
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    
    wav = wav.astype(np.float32)
    return wav


def cut_wav(L, wav):
    samples = len(wav)
    if samples < L:
        wav = np.pad(wav, (0, L - samples), 'constant', constant_values=0.0)
    else:
        start = random.randint(0, samples - L)
        wav = wav[start:start + L]

    return wav


def norm_wav(wav):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    return wav / np.max( np.abs(wav) )


def trim_wav(wav, threshold=0.01):
    assert isinstance(wav, np.ndarray) and len(wav.shape)==1, 'Wav file should be 1D numpy array'
    cut = np.where((abs(wav)>threshold))[0]
    wav = wav[cut[0]:(cut[-1]+1)]
    return wav