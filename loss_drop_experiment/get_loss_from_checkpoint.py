import argparse
from datasets.wavloader import create_dataloader
from model.tier import Tier
from model.tts import TTS
import torch
import torch.nn as nn



# INPUT
def get_checkpoint_loss(chkpt_path):
  tier = int(chkpt_path.split('tier')[1].split('_')[0]) # Yes it's hacky, sorry
  checkpoint = torch.load(chkpt_path)
  hp = checkpoint['hp_str']
  model = get_model(tier, hp)
  model.load_state_dict(checkpoint['model'])
  optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=hp.train.adam.lr
  )
  args = get_args(tier)
  testloader = get_testloader(hp, args)
  loss = compute_loss(args, model, testloader, criterion)
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
    model = nn.DataParallel(model).cuda()
    return model

def get_testloader(hp, args):
  testloader = create_dataloader(hp, args, train=False)

def compute_loss(args, model, testloader, criterion):
    model.eval()
    # torch.backends.cudnn.benchmark = False

    test_loss = []
    loader = tqdm(testloader, desc='Testing is in progress', dynamic_ncols=True)
    with torch.no_grad():
        for input_tuple in loader:
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
            loss = criterion(
                target.cuda(non_blocking=True),
                mu, std, pi,
                audio_lengths.cuda(non_blocking=True)
            )
            test_loss.append(loss)

        test_loss = sum(test_loss) / len(test_loss)
    return test_loss