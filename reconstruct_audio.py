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
    filename = os.path.basename(file_list[idx])
    wav = read_wav_np(file_list[idx], sample_rate=hp.audio.sr)
    yield filename, wav

def deconstruct_audio(wav):
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  melgen = MelGen(hp)
  tierutil = TierUtil(hp)
  mel = melgen.get_normalized_mel(wav)
  tier_to_breakdown = {}
  for tier in range(1, 7):
    source, target = tierutil.cut_divide_tiers(mel, tier)
    print("Tier %d has source dims: %s, target dims %s" % (tier, source.shape, target.shape))
    tier_to_breakdown[tier] = (source, target)
  tier_to_breakdown[7] = (mel, mel)
  return tier_to_breakdown

def reconstruct_audio(filename, tier_to_breakdown):
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  melgen = MelGen(hp)
  tierutil = TierUtil(hp)
  final_reconstruction = None

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
    if tier < 6:
      next_tier = tier_to_breakdown[tier+1][0]
      assert (reconstructed_mel == next_tier).all(), "Tier %d not created from Tier %d" % (tier+1, tier)
    else:
      final_reconstruction = reconstructed_mel
  print('reconstructing audio...')
  reconstructed_audio = melgen.reconstruct_audio(final_reconstruction)
  melgen.save_audio('reconstructed_'+filename, reconstructed_audio)




breakdown = None
audio_files = get_audio()
for filename, wav in audio_files:
  breakdown = deconstruct_audio(wav)
  reconstruct_audio(filename, breakdown)
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  melgen = MelGen(hp)
  melgen.save_audio('original_'+filename, wav)
  print('')
  print('')
  break

