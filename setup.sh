sudo apt update -y
sudo apt upgrade -y
sudo apt install -y software-properties-common 
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10 
sudo apt install -y python3.10-venv
python3.10 -m venv scenic_venv
source scenic_venv/bin/activate
git clone https://github.com/google-research/scenic.git
pip install scenic/.
pip uninstall jax jaxlib -y
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install ipdb wandb
rm -rf scenic