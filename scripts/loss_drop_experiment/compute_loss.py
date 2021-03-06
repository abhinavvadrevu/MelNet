import csv
from google_drive_downloader import GoogleDriveDownloader as gdd
from loss_drop_experiment.get_loss_from_checkpoint import get_checkpoint_loss
print("importing complete")

with open('loss_drop_experiment/download_links.csv') as csvfile:
  csv_reader = csv.reader(csvfile)
  header = next(csv_reader)
  print(header)
  for row in csv_reader:
    [filename, _, fileid, _, _, _] = row
    if 'tier1' in filename:
      continue
    print((filename, fileid))
    # Download the weights
    dest_path = './loss_drop_experiment/weights/%s' % filename
    gdd.download_file_from_google_drive(file_id=fileid,
                                    dest_path=dest_path,
                                    unzip=False)
    print("download completed")

    # Compute the loss
    loss = get_checkpoint_loss(dest_path)
    print("loss computed")
    print((filename, fileid))
    print(loss)
    print('')
    break

    # Write the loss down
    # Delete the weights file