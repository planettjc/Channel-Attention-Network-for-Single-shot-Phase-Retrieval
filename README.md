# Network Code for Quantitative Phase Reconstruction in "A Universal Microscope Add-on for Snapshot Quantitative Phase Imaging by Spectral Encoding"

#### Author
Weihang Zhang

The code is modified from the [code](https://github.com/wwlCape/HAN) in [Single Image Super-Resolution via a Holistic Attention Network (ECCV 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_12). We thank the authors for sharing their codes.


## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
cd Phase/src
pip install -r requirements.txt
```
The installation could be finished within an hour on a common Linux server.

## 2. Prepare Dataset:
Put the encoded measurement (suffix: _IN.png) and the ground truth phase (suffix: _PH.png) into the corresponding folders of `Phase/src/dataset`, then recollect them and the code scripts as the following form:

```shell
|--Phase
    |--experiment
        |--phase_model (automatically established after Training)
            |--model
                |--model_latest.pt
            |--results_Val
                |--200000_PH_Recon.png
                |--200001_PH_Recon.png
                ：
            |--config.txt
            |--log.txt
            |--loss.pt
            |--loss_log.pt
            |--optimizer.pt
            |--psnr_log.pt
        |--phase_test_result
            |--results-Test (automatically established after Testing)
                |--100000_PH_Recon.png
                |--100001_PH_Recon.png
                ： 
            |--config.txt
            |--log.txt
            |--Merged_10.tif (automatically generated after Merging)
    |--src
        |--data
            |--__init__.py
            |--benchmark.py
            |--common.py
            |--dataproc.py
            |--phase.py
            |--video.py
        |--dataset
            |--Gt
                |--000000_PH.png
                |--000001_PH.png
                ：  
            |--Input
                |--000000_IN.png
                |--000001_IN.png
                ：  
            |--Test
                |--Gt
                    |--100000_PH.png
                    |--100001_PH.png
                    ：  
                |--Input
                    |--100000_IN.png
                    |--100001_IN.png
                    ：  
            |--Val
                |--Gt
                    |--200000_PH.png
                    |--200001_PH.png
                    ：  
                |--Input
                    |--200000_IN.png
                    |--200001_IN.png
                    ：  
        |--loss
            |--__init__.py
            |--adversarial.py
            |--discriminator.py
            |--vgg.py
        |--model
            |--__init__.py
            |--common.py
            |--han.py
        |--__init__.py
        |--dataloader.py
        |--main.py
        |--merge.py
        |--option.py
        |--requirements.txt
        |--trainer.py
        |--utility.py
    |--README.md
```

## 3. Experiment:

### 3.1　Training

The training data (including the intensity measurement and ground truth phase) should be put into the `Input` and `Gt` subfolders in `Phase/src/dataset` folder (as determined by the `dir_data` parameter below) respectively. The validation set should be put into the `Input` and `Gt` subfolders in `Phase/src/dataset/Val` folder respctively.

```shell
cd Phase/src

python main.py --save phase_model --save_results --patch_size 128 --batch_size 8 --test_every 1000 --n_GPUs 1 --dir_data ./dataset --decay 80 --gamma 0.6  --epochs 800 --reset --chop --scheduler StepLR
```

The training log and trained model will be available in the `model` subfolder in the folder determined by the `save` parameter above. 

The training time depends on the hardware setup and the network parameters. On a single NVIDIA GeForce RTX 3090 GPU, with the aforementioned parameters, the training time for one epoch is within 600 seconds.

### 3.2　Testing	

The testing data (including the measurement and ground truth (if exists) patches) should be put into the `Input` and `Gt` subfolders in `Phase/src/dataset/Test` folder respctively. For convenience, a small realistic dataset acquired by us has been provided.

The parameter `pre_train` should be set to be the path of the trained model. For convenience, a pre-trained model has been provided.

```shell
cd Phase/src

python main.py --data_test Test --pre_train ../experiment/phase_model/model/model_latest.pt --test_only --save phase_test_result --save_results --n_GPUs 1 --dir_data ./dataset

```

The reconstrcuted phase data will be output into the `results-Test` subfolder in the folder determined by the `save` parameter. For convenience, the result on the provided realistic dataset has been provided.

The testing time depends on the hardware setup and the network parameters. On a single NVIDIA GeForce RTX 3090 GPU, with the aforementioned parameters, the processing time for one patch is within 0.2 seconds.

### 3.3　Merging
Generate the phase image from the reconstructed results by using the following script, and the merged image will be output into the folder determined by the `save` parameter in Testing.

```shell
python merge.py
```	




