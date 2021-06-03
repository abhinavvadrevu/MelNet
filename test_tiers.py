import os
import glob
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from trainer import parse_args
from inference import parse_args as parse_inference_args
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np, cut_wav, get_length, process_blizzard
from model.model import MelNet
from utils.audio import MelGen
from utils.gmm import sample_gmm
from utils.hparams import HParam
from utils.plotting import plot_spectrogram_to_numpy
from utils.tierutil import TierUtil
from text import text_to_sequence
from datasets.wavloader import AudioOnlyDataset

WAV_FILE = 'datasets/BC2013_segmented_v0_wav1/jane_eyre/CB-JE-33-71.wav'
SENTENCE = 'Yes. Again came the blank of a pause: the clock struck eight strokes.'
length = 6.630022675736962
timestep = int((length * 22050.0) / 256.0) + 1
tiers = [
  'chkpt/inference/blizzard-alldata-v5_f3690cd_tier1_011.pt',
  'chkpt/inference/blizzard-alldata-v5_f203de9_tier2_018.pt',
  'chkpt/inference/blizzard-alldata-v5_f203de9_tier3_012.pt',
  'chkpt/inference/blizzard-alldata-v5_f203de9_tier4_008.pt',
  'chkpt/inference/blizzard-alldata-v5_t5_f203de9_tier5_001.pt',
  'chkpt/inference/blizzard-alldata-v5_f203de9_tier6_002.pt'
]
TESTING_TIERS = 5

def get_starting_point(tier_to_breakdown):
  tier = 7 - TESTING_TIERS
  source, target = tier_to_breakdown[tier]
  return source

def deconstruct_audio(wav):
  hp = HParam('./config/blizzard_alldata_v5.yaml')
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

def save_audio(filename, final_reconstruction):
  hp = HParam('./config/blizzard_alldata_v5.yaml')
  melgen = MelGen(hp)
  reconstructed_audio = melgen.reconstruct_audio(final_reconstruction)
  melgen.save_audio('temp/reconstructed_'+filename, reconstructed_audio)

# def reconstruct_audio(filename, tier_to_breakdown):
#   hp = HParam('./config/blizzard_alldata_v5.yaml')
#   melgen = MelGen(hp)
#   tierutil = TierUtil(hp)
#   final_reconstruction = None

#   # Verify that tier 2 is conditionally generated from just tier 1
#   assert (breakdown[2][0] == breakdown[1][1]).all(), "Tier 2 not created from Tier 1"

#   for tier in range(2, 7):
#     source = tier_to_breakdown[tier][0]
#     target = tier_to_breakdown[tier][1]
    
#     source_tensor = torch.unsqueeze(torch.from_numpy(source), 0)
#     target_tensor = torch.unsqueeze(torch.from_numpy(target), 0)
#     reconstructed_mel_tensor = tierutil.interleave(source_tensor, target_tensor, tier+1)
#     reconstructed_mel = reconstructed_mel_tensor.numpy()[0]

#     # Verify that interleaving the source and target of the current tier conditionally generates the source of the next tier
#     if tier < 6:
#       next_tier = tier_to_breakdown[tier+1][0]
#       assert (reconstructed_mel == next_tier).all(), "Tier %d not created from Tier %d" % (tier+1, tier)
#     else:
#       final_reconstruction = reconstructed_mel
#   print('reconstructing audio...')
#   reconstructed_audio = melgen.reconstruct_audio(final_reconstruction)
#   melgen.save_audio('reconstructed_'+filename, reconstructed_audio)

def run_inference(source, timestep, tier_to_breakdown):
  # First load in the model
  hp = HParam('./config/blizzard_alldata_v5.yaml')
  infer_hp = HParam('./config/inference.yaml')
  args = parse_inference_args(['-c', 'config/blizzard_alldata_v5.yaml', '-p', 'config/inference.yaml', '-t', str(timestep), '-n', 'test_tiers', '-i', SENTENCE])
  model = MelNet(hp, args, infer_hp).cuda()
  model.load_tiers()
  model.eval()
  audio_lengths = torch.LongTensor([0]).cuda()
  for t in tqdm(range(model.args.timestep // model.t_div)):
    audio_lengths += 1
  ## Tier 2~N ##
  x = torch.unsqueeze(torch.from_numpy(source), 0)
  for tier in tqdm(range(model.hp.model.tier + 1 - TESTING_TIERS, model.hp.model.tier + 1)):
      tqdm.write('Tier %d' % tier)
      # Save original source and inference source
      actual_source = tier_to_breakdown[tier][0]
      actual_target = tier_to_breakdown[tier][1]
      actual_interleaved = tier_to_breakdown[tier+1][0]
      current_source = x
      save_image(x.detach().numpy()[0], 'tier_%d_inference_source' % tier)
      save_image(actual_source, 'tier_%d_actual_source' % tier)
      save_image(actual_target, 'tier_%d_actual_target' % tier)
      mu, std, pi = model.tiers[tier](x, audio_lengths)
      temp = sample_gmm(mu, std, pi)
      save_image(temp[0].cpu().detach().numpy(), 'tier_%d_inference_target' % tier)
      x = model.tierutil.interleave(x, temp, tier + 1)
      save_image(x.detach().numpy()[0], 'tier_%d_inference_interleaved' % tier)
      save_image(actual_interleaved, 'tier_%d_actual_interleaved' % tier)
  reconstructed_mel_tensor = x.detach().numpy()
  return reconstructed_mel_tensor[0]

def save_image(mel, name):
  # print(type(mel))
  # newmel = mel.detach().numpy()[0]
  spectrogram1 = plot_spectrogram_to_numpy(mel)
  plt.imsave(os.path.join('validation_tests', name + '.png'), spectrogram1.transpose((1, 2, 0)))

def save_images(final_mel, target_mel, filename):
  spectrogram1 = plot_spectrogram_to_numpy(final_mel)
  plt.imsave(os.path.join('validation_tests', filename + '_generated.png'), spectrogram1.transpose((1, 2, 0)))
  spectrogram2 = plot_spectrogram_to_numpy(target_mel)
  plt.imsave(os.path.join('validation_tests', filename + '_target.png'), spectrogram2.transpose((1, 2, 0)))

# breakdown = None
# audio_files = get_audio()
# for filename, wav in audio_files:
#   breakdown = deconstruct_audio(wav)
#   reconstruct_audio(filename, breakdown)
#   hp = HParam('./config/blizzard_alldata_v5.yaml')
#   melgen = MelGen(hp)
#   melgen.save_audio('original_'+filename, wav)
#   print('')
#   print('')
#   break

# First deconstruct the wav file
filename = os.path.basename(WAV_FILE)
wav = read_wav_np(WAV_FILE, sample_rate=22050)
tier_to_breakdown = deconstruct_audio(wav)

# Now find the appropriate starting point
source = get_starting_point(tier_to_breakdown)

# Now run inference
final_mel = run_inference(source, timestep, tier_to_breakdown)
target_mel = tier_to_breakdown[7][0]
# save_images(final_mel, target_mel, filename)
save_audio('generated_'+filename, final_mel)
save_audio('target_'+filename, target_mel)