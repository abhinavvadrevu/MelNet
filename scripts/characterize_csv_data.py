import glob
import librosa
import os
import csv
import plotille
from tqdm import tqdm
import audiosegment

with open('datasets/blizzard_train.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    rows = list(reader)

headers = rows.pop(0)

# trainset = [row for row in rows if row[4]]
# testset = [row for row in rows if not row[4]]
trainset = rows

print("Number of txt/wav samples in trainset: %d" % len(trainset))
# print("Number of txt/wav samples in testset: %d" % len(testset))
print("Number of txt/wav samples in total: %d" % len(rows))

# Wav legth histogram
all_lengths = [float(x[3]) for x in trainset]
# print(all_lengths[0:10])
hist1 = plotille.hist(all_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print("Wav lengths in trainset")
print("Total length: %d seconds" % sum(all_lengths))
print("Total length: %d hours" % (sum(all_lengths) / 3600.0))
print(hist1)
print('')

lengths_under_10 = [x for x in all_lengths if (x < 10 and x > 0.5)]
hist2 = plotille.hist(lengths_under_10, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print("Wav lengths in trainset for length < 10s")
print("Total length for under 10s: %d seconds" % sum(lengths_under_10))
print("Total length for under 10s: %d hours" % (sum(lengths_under_10) / 3600.0))
print(hist2)
print('')

lengths_under_6 = [x for x in all_lengths if x < 6]
hist3 = plotille.hist(lengths_under_6, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
print("Wav lengths in trainset for length < 6s")
print("Total length for under 6s: %d" % (sum(lengths_under_6) / 3600.0))
print(hist3)
print('')


# # Text character lengths histogram
# sentence_lengths = list(map(len, new_sentences))
# old_sentences = list(map(lambda x:x[1], old_dataset))
# old_sentence_lengths = list(map(len, old_sentences))

# hist1 = plotille.hist(sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
# print(hist1)
# hist2 = plotille.hist(old_sentence_lengths, bins=40, width=80, log_scale=False, linesep='\n', lc=None, bg=None, color_mode='names')
# print(hist2)

# # Max text length
# print("Max txt length: %f" % max(sentence_lengths))
# print("Max old txt length: %f" % max(old_sentence_lengths))

# # Min text length
# print("Min txt length: %f" % min(sentence_lengths))
# print("Min old txt length: %f" % min(old_sentence_lengths))


