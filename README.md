# Human Pose Prediction
## Description
This is the library implementation of the human pose prediction using SeSGCN model. The [CHICO-PoseForecasing](https://github.com/AlessioSam/CHICO-PoseForecasting.git) repository which has the official implementation of the model is used as a reference in this implementation. It uses the proposed Seperable-Sparse Graph Convolution Network (SeSGCN) to predict the human pose by feeding it the recent observed poses. 

## Getting Started
### Prerequisites
* `python 3.8`
* `pytorch 1.12`
* `cuda 10.2+` (recommeded for gpu support but optional)

### Setting up using Conda
```bash
# Setup Conda Environment
conda create --name <my-env> -python=3.8 # replace <my-env> with a your environment name
conda activate <my-env>
conda install pytorch torchvision torchaudio -c pytorch

# Clone the repository
git clone https://github.com/OmkarKabadagi5823/Human-Pose-Prediction.git
```

### Running test scripts
TODO
 

## Documentation
TODO