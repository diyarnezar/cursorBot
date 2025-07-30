#!/bin/bash
# Setup script for Amazon Linux 2 to configure environment for trading pipeline
# This script will install Python 3.10+, development tools, required libraries,
# and set up a virtual environment with all necessary packages

# Exit on error
set -e

echo "===================================="
echo "Setting up trading pipeline environment"
echo "===================================="

# Create a log file
LOG_FILE="setup_log.txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "Setup started at: $(date)"
echo "Installing system packages..."

# Update system packages
sudo yum update -y

# Install development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget git cmake3 openssl-devel bzip2-devel libffi-devel

# Install Python 3.10 (Amazon Linux 2 comes with older versions)
echo "Installing Python 3.10..."
cd /opt
sudo wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
sudo tar xzf Python-3.10.12.tgz
cd Python-3.10.12
sudo ./configure --enable-optimizations
sudo make altinstall
cd /opt
sudo rm -f Python-3.10.12.tgz

# Create symlinks for python3.10
sudo ln -sf /usr/local/bin/python3.10 /usr/bin/python3
sudo ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

# Verify Python installation
echo "Python version:"
python3 --version
pip3 --version

# Install system libraries needed for numerical computation
echo "Installing numerical computation libraries..."
sudo yum install -y atlas-devel lapack-devel

# Install libraries needed for TA-Lib
sudo yum install -y gcc gcc-c++ make wget

# Install TA-Lib
echo "Installing TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Set up LD_LIBRARY_PATH for TA-Lib
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Create project directory and virtual environment
echo "Setting up project directory and virtual environment..."
PROJECT_DIR="$HOME/trading_pipeline"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TA-Lib Python wrapper
echo "Installing TA-Lib Python wrapper..."
pip install numpy  # Required before TA-Lib
pip install ta-lib

# Install required Python packages
echo "Installing required Python packages..."
pip install numpy pandas matplotlib scipy scikit-learn statsmodels
pip install pandas-ta  # Additional TA functions
pip install ccxt  # For cryptocurrency exchange API
pip install gymnasium stable-baselines3  # For RL environment
pip install optuna  # For hyperparameter optimization
pip install lightgbm xgboost  # For gradient boosting models
pip install tensorflow  # For transformer models
pip install tensorboard  # For visualizing training
pip install hmmlearn  # For HMM models
pip install joblib  # For model persistence
pip install pymongo  # For database connection
pip install pytz  # For timezone handling
pip install requests  # For API calls
pip install plotly dash  # For interactive visualizations
pip install pytest  # For testing

# Optional: Configure CUDA for GPU acceleration
echo "Do you want to install NVIDIA drivers and CUDA? (y/n)"
read -r install_cuda

if [[ $install_cuda == "y" || $install_cuda == "Y" ]]; then
    echo "Installing NVIDIA drivers and CUDA..."
    # Add NVIDIA repository
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    
    # Install NVIDIA driver and CUDA
    sudo yum clean all
    sudo yum -y install nvidia-driver-latest-dkms
    sudo yum -y install cuda
    
    # Set up environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Install cuDNN
    echo "Please download cuDNN manually from NVIDIA website and install it"
    echo "See: https://developer.nvidia.com/cudnn"
    
    # Verify CUDA installation
    source ~/.bashrc
    nvidia-smi
    nvcc --version
fi

# Create a README file with instructions
cat > $PROJECT_DIR/README.md << 'EOF'
# Trading Pipeline Environment

This environment is set up with all necessary dependencies for the trading pipeline.

## Activation

To activate the virtual environment:

```bash
cd ~/trading_pipeline
source venv/bin/activate
```

## Python Packages Installed

- numpy, pandas, matplotlib: Data manipulation and visualization
- scikit-learn, lightgbm, xgboost: Machine learning models
- tensorflow: Deep learning (transformer models)
- gymnasium, stable-baselines3: Reinforcement learning
- ta-lib, pandas-ta: Technical analysis indicators
- optuna: Hyperparameter optimization
- ccxt: Cryptocurrency exchange API
- And more...

## Running the Pipeline

Copy your trading pipeline code to this directory and run it within the virtual environment.
EOF

echo "===================================="
echo "Setup completed successfully!"
echo "===================================="
echo "To activate the environment, run:"
echo "cd $PROJECT_DIR && source venv/bin/activate"
echo ""
echo "See $PROJECT_DIR/README.md for more information."
echo "Setup finished at: $(date)"

# Add a reminder to copy project files
echo "Remember to copy your project files to $PROJECT_DIR before running."
