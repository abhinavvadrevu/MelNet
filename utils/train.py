import glob
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import traceback

from tqdm import tqdm

from datasets.wavloader import create_dataloader
from model.tier import Tier
from model.tts import TTS
from model.loss import GMMLoss
from .utils import get_commit_hash
from .audio import MelGen
from .tierutil import TierUtil
from .constant import f_div, t_div
from .validation import validate

def train(args, pt_dir, chkpt_path, writer, logger, hp, hp_str):
    # If the train-all flag is set, train all tiers one-by-one. Otherwise, just train one tier.
    # if the train-all flag is set, start training at the tier specified, and work down from there.
    if args.train_all:
        num_tiers = hp['model']['tier']
        cur_tier = args.tier
        main_name = args.name
        i = 1
        while True:
            print("Training tier #%d" % cur_tier)
            if cur_tier == 1:
                args.tts = True
            else:
                args.tts = False
            args.tier = cur_tier
            print("Beginning training tier %d, for %s with tts=%s. This is iteration #%d." % (args.tier, args.name, str(args.tts), i))
            trainloader = create_dataloader(hp, args, train=True)
            testloader = create_dataloader(hp, args, train=False)
            train_helper(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str)
            cur_tier -= 1
            if cur_tier == 0:
                print("All tiers were trained an epoch! Starting again at top tier.")
                cur_tier = num_tiers
            print('')
            i += 1
    else:
        print("Training a specific tier")
        trainloader = create_dataloader(hp, args, train=True)
        testloader = create_dataloader(hp, args, train=False)
        train_helper(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str)

def train_helper(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    if args.tts:
        model = TTS(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[args.tier],
            layers=hp.model.layers[args.tier-1]
        )
    else:
        model = Tier(
            hp=hp,
            freq=hp.audio.n_mels // f_div[hp.model.tier+1] * f_div[args.tier],
            layers=hp.model.layers[args.tier-1],
            tierN=args.tier
        )
    model = nn.DataParallel(model).cuda()
    melgen = MelGen(hp)
    tierutil = TierUtil(hp)
    criterion = GMMLoss()

    if hp.train.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=hp.train.rmsprop.lr, 
            momentum=hp.train.rmsprop.momentum
        )
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hp.train.adam.lr
        )
    elif hp.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=hp.train.sgd.lr
        )
    else:
        raise Exception("%s optimizer not supported yet" % hp.train.optimizer)

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    file_pattern = os.path.join(pt_dir, '%s_*_tier%d_[0-9][0-9][0-9].pt' % (args.name, args.tier))
    files = glob.glob(file_pattern)
    if len(files) > 0:
        last_epoch = max(map(lambda x: int(x[-6:-3]), files))
        last_epoch_str = f'{last_epoch:03}'
        new_filepattern = os.path.join(pt_dir, '%s_*_tier%d_%s.pt' % (args.name, args.tier, last_epoch_str))
        last_file = glob.glob(file_pattern)[-1]
        last_githash = last_file[-20:-13]
        chkpt_file_path = '%s_%s_tier%d_%03d.pt' % (args.name, last_githash, args.tier, last_epoch)
        chkpt_path = os.path.join(pt_dir, chkpt_file_path)

    if chkpt_path is not None:
        print("Resuming training from checkpoint: %s" % chkpt_path)
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

        githash = checkpoint['githash']
    else:
        logger.info("Starting new training run.")

    # use this only if input size is always consistent.
    # torch.backends.cudnn.benchmark = True
    try:
        model.train()
        optimizer.zero_grad()
        loss_sum = 0

        epochs_to_train_on = itertools.count(init_epoch + 1)
        if args.train_all:
            epochs_to_train_on = [init_epoch + 1]

        for epoch in epochs_to_train_on:
            loader = tqdm(trainloader, desc='Train data loader', dynamic_ncols=True)
            for input_tuple in loader:
                if args.tts:
                    seq, text_lengths, source, target, audio_lengths = input_tuple
                    mu, std, pi, _ = model(
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
                step += 1
                (loss / hp.train.update_interval).backward()
                loss_sum += loss.item() / hp.train.update_interval

                if step % hp.train.update_interval == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if step % hp.log.summary_interval == 0:
                        writer.log_training(loss_sum, step)
                        loader.set_description("Loss %.04f at step %d" % (loss_sum, step))
                    loss_sum = 0

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.04f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

            save_path = os.path.join(pt_dir, '%s_%s_tier%d_%03d.pt'
            % (args.name, githash, args.tier, epoch))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
                'githash': githash,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

            validate(args, model, melgen, tierutil, testloader, criterion, writer, step)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
