mkdir workspace
cd workspace
git clone https://github.com/abhinavvadrevu/MelNet.git
cd MelNet/datasets
curl https://blizzard2013.s3.amazonaws.com/blizzard2013/lessac/segmented_training_data.zip -O .
unzip segmented_training_data.zip
cd ..
sudo apt update
sudo apt install ffmpeg
screen
source activate pytorch_p36
pip install -r requirements.txt
echo "Run the script now. Eg:"
echo "python trainer.py -c ./config/blizzard.yaml -n blizzard_t6_b2 -t 6 -b 2 -s TTS"