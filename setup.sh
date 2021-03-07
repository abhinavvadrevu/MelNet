mkdir workspace
cd workspace
git clone https://github.com/abhinavvadrevu/MelNet.git
cd MelNet/datasets
curl https://blizzard2013.s3.amazonaws.com/blizzard2013/lessac/segmented_compressed.zip -O .
unzip segmented_compressed.zip
cd ..
sudo apt update
# sudo killall apt apt-get
sudo apt install ffmpeg
# screen
# source activate pytorch_p36
# pip install -r requirements.txt
# echo "Run the script now. Eg:"
# echo "python trainer.py -c ./config/blizzard-5-tier.yaml -n blizzard_t5 -t 5 -b 1 -s TTS"