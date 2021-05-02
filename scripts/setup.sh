# mkdir .aws
# touch .aws/credentials
mkdir workspace
cd workspace
git clone https://github.com/abhinavvadrevu/MelNet.git
cd MelNet/datasets
curl https://blizzard2013.s3.amazonaws.com/blizzard2013/lessac/segmented_compressed.zip -O
unzip segmented_compressed.zip
# curl https://blizzard2013.s3.amazonaws.com/blizzard-compressed-6-tiers.zip -O
# unzip segmented_compressed.zip
cd ..
sudo apt update
# sudo killall apt apt-get
sudo apt install ffmpeg
# screen
# source activate pytorch_p36
pip install --upgrade pip
pip install -r requirements.txt

# Download all the latest checkpoints in the right places
mkdir chkpt
mkdir chkpt/blizzard-compressed-12layers-t1
cd chkpt/blizzard-compressed-12layers-t1
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t1/blizzard-compressed-12layers-t1_3790578_tier1_050.pt -O
cd ../..

mkdir chkpt/blizzard-compressed-12layers-t2
cd chkpt/blizzard-compressed-12layers-t2
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t2/blizzard-compressed-12layers-t2_7f636a8_tier2_044.pt -O
cd ../..

mkdir chkpt/blizzard-compressed-12layers-t3
cd chkpt/blizzard-compressed-12layers-t3
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t3/blizzard-compressed-12layers-t3_3790578_tier3_014.pt -O
cd ../..

mkdir chkpt/blizzard-compressed-12layers-t4
cd chkpt/blizzard-compressed-12layers-t4
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t4/blizzard-compressed-12layers-t4_3790578_tier4_019.pt -O
cd ../..

mkdir chkpt/blizzard-compressed-12layers-t5
cd chkpt/blizzard-compressed-12layers-t5
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t5/blizzard-compressed-12layers-t5_3790578_tier5_034.pt -O
cd ../..

mkdir chkpt/blizzard-compressed-12layers-t6
cd chkpt/blizzard-compressed-12layers-t6
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/chkpt/blizzard-compressed-12layers-t6/blizzard-compressed-12layers-t6_3790578_tier6_014.pt -O
cd ../..



mkdir logs
cd logs
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/logs/blizzard-compressed-12layers-t1/blizzard-compressed-12layers-t1-1618761441.log -O
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/logs/blizzard-compressed-12layers-t1/blizzard-compressed-12layers-t1-1618770883.log -O
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/logs/blizzard-compressed-12layers-t1/events.out.tfevents.1618761441.chronicle-a100.5519.0 -O
curl https://melnet-training-runs.s3.amazonaws.com/blizzard-compressed/logs/blizzard-compressed-12layers-t1/events.out.tfevents.1618770883.chronicle-a100.12205.0 -O
cd ..

echo "Run the script now. Eg:"
echo "python trainer.py -c config/blizzard_compressed.yaml -n blizzard-compressed-12layers-t1 -t 1 -b 1 -s TTS"
echo "Watch GPU memory with:"
echo "gpustat -cpi"