import librosa
import glob

filelist = glob.glob('datasets/complete_blizzard/train_wav/*')
length = 0.0
for file in filelist:
  length += librosa.get_duration(filename=file)

print("Total length in seconds: %s" % str(length))
print("Total length in hours: %s" % str(length / 60.0 / 60.0))