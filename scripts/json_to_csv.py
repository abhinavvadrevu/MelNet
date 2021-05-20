import json
import csv
from tqdm import tqdm
import random


# Opening JSON file and loading the data
# into the variable data
with open('datasets/blizzard2.json') as json_file:
    data = json.load(json_file)
  
# now we will open a file for writing
train_data_file = open('datasets/blizzard_train.csv', 'w')
test_data_file = open('datasets/blizzard_test.csv', 'w')
  
# create the csv writer object
train_csv_writer = csv.writer(train_data_file)
test_csv_writer = csv.writer(test_data_file)
  
# Counter variable used for writing 
# headers to the CSV file
count = 0
  
for data_to_save in tqdm(data):
    del(data_to_save['train'])
    
    if count == 0:
        # Writing headers of CSV file
        headers = data_to_save.keys()
        train_csv_writer.writerow(headers)
        test_csv_writer.writerow(headers)
        count += 1
    
    if 'CB-20K1-17-59' in data_to_save['wav_path']:
      data_to_save['parsed_sentence'] = 'It crossed the equator December first in one hundred forty two degrees long;~'
    if 'CB-20K2-14-42' in data_to_save['wav_path']:
      data_to_save['parsed_sentence'] = 'We know that in those Antarctic countries James Ross found two craters the Erebus and Terror in full activity on the one hundred and sixty seventh meridian latitude seventy seven degrees 32 minutes.~'
    if 'CB-CAR-05-13' in data_to_save['wav_path']:
      data_to_save['parsed_sentence'] = 'In one corner at the top of it is the name as well as I could read Marcia Karnstein and the date sixteen ninety;~'

    # Writing data of CSV file
    train = False if random.random() < 0.05 else True
    values = data_to_save.values()
    if train:
      train_csv_writer.writerow(values)
    else:
      test_csv_writer.writerow(values)
  
train_data_file.close()
test_data_file.close()