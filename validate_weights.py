import os
import glob
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from trainer import parse_args
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from inference import parse_args as parse_inference_args
from trainer import parse_args as parse_train_args
from utils.utils import read_wav_np, cut_wav, get_length, process_blizzard
from utils.audio import MelGen
from utils.constant import t_div
from utils.hparams import HParam
from utils.tierutil import TierUtil
from text import text_to_sequence
from datasets.wavloader import AudioOnlyDataset


# load up the test set
def load_testset(tier):
  args = parse_train_args(['-c', './config/blizzard_compressed_experiments.yaml', '-n', 'blizzard_compressed_validation', '-t', str(tier), '-b', '1'])
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  
  dataset = []
  with open(os.path.join(hp.data.path, 'prompts.gui'), 'r') as f:
    lines = f.read().splitlines()
    filenames = lines[::3]
    sentences = lines[1::3]
    for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
        wav_path = os.path.join(self.root_dir, 'wavn', filename + '.wav')
        length = get_length(wav_path, hp.audio.sr)
        if length < hp.audio.duration:
            dataset.append((wav_path, sentence))
  random.seed(123)
  random.shuffle(dataset)
  dataset = dataset[int(0.95 * len(dataset)):]

  for i in range(len(dataset)):
    text = dataset[i][1]
    wav = read_wav_np(dataset[i][0], sample_rate=self.hp.audio.sr)
    filename = os.path.basename(dataset[i][0])
    yield filename, text, wav

# break the wav into mel tiers
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

def get_timestep(wav):
  hp = HParam('./config/blizzard_compressed_experiments.yaml')
  hop_length = hp.audio.hop_length
  frames = len(wav)
  return int(float(frames) / float(hop_length)) + 1

# run inference on text
def inference(text, timestep=64):
  args = parse_inference_args(['-c', 'config/blizzard_compressed_experiments.yaml', '-p', 'config/inference.yaml', '-t', str(timestep), '-n', 'hw_blizzard_compressed', '-i', text])
  hp = HParam('./config/blizzard_compressed_experiments.yaml')

  assert timestep % t_div[hp.model.tier] == 0, \
      "timestep should be divisible by %d, got %d" % (t_div[hp.model.tier], timestep)

  model = MelNet(hp, args, hp).cuda()
  model.load_tiers()
  model.eval()

  with torch.no_grad():
      generated = model.sample(args.input)
      # breakdown, generated = sample_model_with_breakdown(model, args.input)

  melspec = generated[0]
  return melspec

def sample_model_with_breakdown(model, condition):
  x = None
  seq = torch.from_numpy(process_blizzard(condition)).long().unsqueeze(0)
  input_lengths = torch.LongTensor([seq[0].shape[0]]).cuda()
  audio_lengths = torch.LongTensor([0]).cuda()
  breakdown = {}

  ## Tier 1 ##
  tqdm.write('Tier 1')
  for t in tqdm(range(model.args.timestep // model.t_div)):
      audio_lengths += 1
      if x is None:
          x = torch.zeros((1, model.n_mels // model.f_div, 1)).cuda()
      else:
          x = torch.cat([x, torch.zeros((1, model.n_mels // model.f_div, 1)).cuda()], dim=-1)
      for m in tqdm(range(model.n_mels // model.f_div)):
          torch.cuda.synchronize()
          if model.infer_hp.conditional:
              mu, std, pi, _ = model.tiers[1](x, seq, input_lengths, audio_lengths)
          else:
              mu, std, pi = model.tiers[1](x, audio_lengths)
          temp = sample_gmm(mu, std, pi)
          x[:, m, t] = temp[:, m, t]
  breakdown[1] = (x.copy(), x.copy())

  ## Tier 2~N ##
  for tier in tqdm(range(2, model.hp.model.tier + 1)):
      tqdm.write('Tier %d' % tier)
      mu, std, pi = model.tiers[tier](x, audio_lengths)
      temp = sample_gmm(mu, std, pi)
      breakdown[tier] = (x.copy(), temp.copy())
      x = model.tierutil.interleave(x, temp, tier + 1)

  return breakdown, x

def save_image(filename, img):
  filename = os.path.basename(filename)
  save_dir = 'validation_tests'
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  print('Saving %s' % filename)
  plt.imsave(os.path.join('validation_tests', filename), img)

for filename, text, wav in load_testset():
  breakdown = deconstruct_audio(wav)
  melspec = breakdown[7][0]
  save_image('original_%s.png' % filename, melspec)
  timestep = get_timestep(wav)
  inferred = inference(text, timestep)
  save_image('final_inferred_%s.png' % filename, inferred)
  