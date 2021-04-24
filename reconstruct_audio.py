import os
import glob
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from trainer import parse_args
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np, cut_wav, get_length, process_blizzard
from utils.audio import MelGen
from utils.hparams import HParam
from utils.tierutil import TierUtil
from text import text_to_sequence
from datasets.wavloader import AudioOnlyDataset

# create testloader
def get_audio():
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  file_list = glob.glob(
    os.path.join(hp.data.path, '**', hp.data.extension),
    recursive=True
  )
  random.seed(123)
  random.shuffle(file_list)
  file_list = file_list[int(0.95 * len(file_list)):]
  for idx in range(len(file_list)):
    wav = read_wav_np(file_list[idx], sample_rate=hp.audio.sr)
    yield wav

def deconstruct_audio(wav):
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  melgen = MelGen(hp)
  tierutil = TierUtil(hp)
  mel = melgen.get_normalized_mel(wav)
  tier_to_breakdown = {}
  for tier in range(1, 7):
    source, target = tierutil.cut_divide_tiers(mel, tier)
    print("Tier %d has source dims: %s, target dims %s" % (tier, source.shape, target.shape))
    # print("Tier %d source: %s" % (tier, source))
    # print("Tier %d target: %s" % (tier, target))
    # print('')
    tier_to_breakdown[tier] = (source, target)
  tier_to_breakdown[7] = (mel, mel)
  return tier_to_breakdown
    # TODO: Run this and check if the source of a given tier is the same as the target from the tier above.

def reconstruct_audio(tier_to_breakdown):
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  melgen = MelGen(hp)
  tierutil = TierUtil(hp)

  # Verify that tier 2 is conditionally generated from just tier 1
  assert (breakdown[2][0] == breakdown[1][1]).all(), "Tier 2 not created from Tier 1"

  for tier in range(2, 7):
    source = tier_to_breakdown[tier][0]
    target = tier_to_breakdown[tier][1]
    
    source_tensor = torch.unsqueeze(torch.from_numpy(source), 0)
    target_tensor = torch.unsqueeze(torch.from_numpy(target), 0)
    reconstructed_mel_tensor = tierutil.interleave(source_tensor, target_tensor, tier+1)
    reconstructed_mel = reconstructed_mel_tensor.numpy()[0]

    # Verify that interleaving the source and target of the current tier conditionally generates the source of the next tier
    next_tier = tier_to_breakdown[tier+1][0]
    assert (reconstructed_mel == next_tier).all(), "Tier %d not created from Tier %d" % (tier+1, tier)

    print('Tier %d shapes: [source, target, reconstruction], [%s, %s, %s]' % (
        tier,
        source.shape,
        target.shape,
        reconstructed_mel.shape,
        ))
  reconstructed_audio = melgen.reconstruct_audio(breakdown[7][0])
  melgen.save_audio('reconstructed_'+filename, reconstructed_audio)



breakdown = None
audio_files = get_audio()
for wav in audio_files:
  breakdown = deconstruct_audio(wav)
  # reconstruct_audio(breakdown)
  break

# def get_dataset(tier):
#   args = parse_args(['-c', './config/blizzard_compressed.yaml', '-n', 'blizzard_compressed_test', '-t', str(tier), '-b', '1'])
#   hp = HParam(args.config)
#   if tier == 1:
#     args = parse_args(['-c', './config/blizzard_compressed.yaml', '-n', 'blizzard_compressed_test', '-t', '1', '-b', '1', '-s', 'TTS'])
#     hp = HParam(args.config)
#   dataset = AudioOnlyDataset(hp, args, False)
#   return dataset

# # deconstruct audio into mel tiers

# def deconstruct_tier(tier):
#   loader = tqdm(create_testloader(tier), desc='Test data loader', dynamic_ncols=True)
#   for input_tuple in loader:
#     if args.tts:
#       seq, text_lengths, source, target, audio_lengths = input_tuple
#     else:
#       source, target, audio_lengths = input_tuple
    
# reconstruct it again