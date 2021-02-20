import requests
from urllib.parse import urlparse, urljoin
from pathlib import Path
import os

paths = [
  # 'https://data.cstr.ed.ac.uk/blizzard2013/lessac/README_for_Lessac_Blizzard2013_CatherineByers_train',
  # 'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_txt1.zip',
  # 'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_txt2.zip',
  # 'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_wav1.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_wav2.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_labels_EM.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_transcripts_selection.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_txt_selection.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_wav_selection.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/BlackBeauty.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/Lessac_Blizzard2013_CatherineByers_train.tar.bz2',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield1.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield2.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield3.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice1.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice2.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice3.zip',
  'https://data.cstr.ed.ac.uk/blizzard2013/lessac/training_inventory.xls'
]

for path in paths:
  print("Downloading %s" % path)
  dir_path = os.path.join('./datasets', urlparse(path).path[1::])
  Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
  resp = requests.get(path, allow_redirects=True, auth=('abhinavvadrevu1@gmail.com', 'FHe6f1xfXcfoH'))
  open(dir_path, 'wb').write(resp.content)
  print("%s was successfully downloaded" % path)