import os
import sys
import time
import logging
import argparse
import platform

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.wavloader import create_dataloader

def parse_args(args):
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
    return parser.parse_args(args)

if __name__ == '__main__':
    print(sys.argv[1:])
    args = parse_args(sys.argv[1:])

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())
    if platform.system() == 'Windows':
        hp.train.num_workers = 0

    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    if not os.path.isdir(hp.log.log_dir):
        os.mkdir(hp.log.log_dir)
    if not os.path.isdir(hp.log.chkpt_dir):
        os.mkdir(hp.log.chkpt_dir)
    if not os.path.isdir(pt_dir):
        os.mkdir(pt_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    assert hp.data.path != '', \
        'hp.data.path cannot be empty: please fill out your dataset\'s path in configuration yaml file.'

    train(args, pt_dir, args.checkpoint_path, writer, logger, hp, hp_str)
