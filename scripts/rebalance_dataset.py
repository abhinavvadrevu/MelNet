import os
import shutil

trainset_index = 45325
testset_index = 567

while trainset_index < 47143:
  train_txt_path = f'datasets/cleaned_blizzard/train_txt/{trainset_index:05d}.txt'
  train_wav_path = f'datasets/cleaned_blizzard/train_wav/{trainset_index:05d}.wav'
  test_txt_path = f'datasets/cleaned_blizzard/test_txt/{testset_index:05d}.txt'
  test_wav_path = f'datasets/cleaned_blizzard/test_wav/{testset_index:05d}.wav'

  os.rename(train_txt_path, test_txt_path)
  os.rename(train_wav_path, test_wav_path)

  trainset_index += 1
  testset_index += 1