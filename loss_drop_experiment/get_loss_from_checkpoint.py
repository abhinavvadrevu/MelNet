import argparse
from datasets.wavloader import create_dataloader
from model.tier import Tier
from model.tts import TTS
import torch
import torch.nn as nn
from utils.hparams import HParam
from utils.constant import f_div, t_div



# INPUT
def get_checkpoint_loss(chkpt_path):
  config_path = 'config/blizzard.yaml'
  tier = int(chkpt_path.split('tier')[1].split('_')[0]) # Yes it's hacky, sorry
  print('tier: %d' % tier)
  checkpoint = torch.load(chkpt_path)
  print("Checkpoint loaded")
  hp = HParam(config_path)
  with open(config_path, 'r') as f:
      model_hp = checkpoint['hp_str']
      hp_str = ''.join(f.readlines())
      if model_hp != hp_str:
        print(model_hp)
        print('')
        print(hp_str)
        print("ERROR: ISSUE WITH DIFFERENT HPs")
  model = get_model(tier, hp)
  print("Got model")
  model.load_state_dict(checkpoint['model'])
  print("Model loaded")
  optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=hp.train.adam.lr
  )
  print("Got optimizer")
  args = get_args(tier)
  print("Got args")
  testloader = get_testloader(hp, args)
  print("Got testloader")
  loss = compute_loss(args, model, testloader, criterion)
  print("Got loss")
  return loss

def get_args(tier):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, required=True,
                      help="yaml file for configuration")
  parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                      help="path of checkpoint pt file to resume training")
  parser.add_argument('-n', '--name', type=str, required=True,
                      help="name of the model for logging, saving checkpoint")
  parser.add_argument('-t', '--tier', type=int, required=True,
                      help="Number of tier to train")
  parser.add_argument('-b', '--batch_size', type=int, required=True,
                      help="Batch size")
  parser.add_argument('-s', '--tts', type=bool, default=False, required=False,
                      help="TTS")
  parser.add_argument('-a', '--train-all', type=bool, default=False, required=False,
                      help="Use this param to train all tiers of a TTS model")
  args = parser.parse_args()
  args.config = '../config/blizzard.yaml'
  args.tier = tier
  args.batch_size = 4
  args.tts = True if tier == 1 else False
  return args

def get_testloader(hp, args):
  return create_dataloader(hp, args, train=False)

def get_model(tier, hp):
    if tier == 1:
        model = TTS(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[tier],
            layers=hp.model.layers[tier-1]
        )
    else:
        model = Tier(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[tier],
            layers=hp.model.layers[tier-1],
            tierN=tier
        )
    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    return model

def get_testloader(hp, args):
  testloader = create_dataloader(hp, args, train=False)

def compute_loss(args, model, testloader, criterion):
    model.eval()
    print("model evaled")
    # torch.backends.cudnn.benchmark = False

    test_loss = []
    loader = tqdm(testloader, desc='Testing is in progress', dynamic_ncols=True)
    with torch.no_grad():
        for input_tuple in loader:
            print("loading inputs")
            if args.tts:
                seq, text_lengths, source, target, audio_lengths = input_tuple
                mu, std, pi, alignment = model(
                    source.cuda(non_blocking=True),
                    seq.cuda(non_blocking=True),
                    text_lengths.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            else:
                source, target, audio_lengths = input_tuple
                mu, std, pi = model(
                    source.cuda(non_blocking=True),
                    audio_lengths.cuda(non_blocking=True)
                )
            print("computing things")
            loss = criterion(
                target.cuda(non_blocking=True),
                mu, std, pi,
                audio_lengths.cuda(non_blocking=True)
            )
            print("computed loss")
            test_loss.append(loss)

        test_loss = sum(test_loss) / len(test_loss)
    return test_loss