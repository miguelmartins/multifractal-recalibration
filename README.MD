[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
# Monofractal and Multifractal Recalibration of Fully Convolutional Networks for Medical Image Segmentation

We propose two attention functions that capture cross-channel statistical scaling regularities:
Monofractal and Multifractal recalibration. 
We build an experimental framework centered around the U-Net and show that these approaches, 
specially Multifractal recalibration, lead to substantial improvements over a baseline augmented
with none or with other attention functions that may also describe each channel such as scSE [1, 2], Style Based Recalibration (SRM)[3], and 
Frequency Channel Attention (FCA) [4]. 
Our experiments cover three public datasets from diverse modalities: ISIC18 (dermoscopy) [5], Kvasir-SEG (endoscopy) [6], and BUSI (ultrasound) [7].  

## Experimental Results 
This table collects the results of the experiments (DICE score) in:  `experiments/isic18`, `experiments/kvasir`, and `experiments/busi`.

| Model                           | ISIC18                            | Kvasir-SEG                       | BUSI                              |
|---------------------------------|-----------------------------------|----------------------------------|-----------------------------------|
| U-Net                           | 85.40 ± 0.25                      | 72.22 ± 1.82                     | 62.20 ± 2.40                      |
| +cSE                            | 85.94 ± 0.36<sup>†</sup>          | 72.72 ± 1.52                     | 65.36 ± 1.36                      |
| +scSE                           | 85.92 ± 0.29<sup>†</sup>          | 72.94 ± 1.07                     | 64.82 ± 1.03                      |
| +SRM                            | 84.33 ± 1.27                      | 61.13 ± 3.42                     | 68.09 ± 3.14<sup>†</sup>          |
| +FCA                            | 86.19 ± 0.75                      | 70.00 ± 2.51                     | 66.27 ± 2.48                      |
| +Mono (ours)                    | 86.24 ± 0.27<sup>‡</sup>          | 71.86 ± 2.37                     | **69.00 ± 2.53<sup>‡</sup>**      |
| +Multi (ours)                   | **86.26 ± 0.28<sup>‡</sup>**      | **74.76 ± 2.20<sup>†</sup>**     | 66.94 ± 2.45<sup>†</sup>          |

<sup>†</sup> The null-hypothesis of the pairwise t-test with regards to the U-Net baseline is rejected with p ≤ 0.05.

<sup>‡</sup> The null-hypothesis of the pairwise t-test with regards to the U-Net baseline is rejected with p ≤ 0.01.

**Bold** indicates the best mean results.



## Recommended requirements
1. Use anaconda/miniconda to create a __python 3.10.9__ virtual environment:
    ```zsh
    $ conda create --name env_name python=3.10.9
    ```
2. Activate environment and update pip:
    ```zsh
    $ (env_name) python3 -m pip install --upgrade pip
    ```
4. Use pip to install packages in `requirements.txt` file:
    ```zsh
    $ (env_name) pip install -r /path/to/project/requirements.txt
    ```
   
Note that this code was developed using Tensorflow 2.14. 

## Dataset files
The exact files used for these experiments can be found in [our OSF repository](https://osf.io/y4djz/?view_only=8eeba61afd104c678558c0da5a82b499).

Note that for ISIC18 we used the [public implementation](https://github.com/NITR098/Awesome-U-Net) of [8] to get the numpy files.

One can simply get the original files of each dataset, and place them in the `/path/to/project/datasets` directory. 

Note that the scripts in `experiments` are expecting the following file and directory structure.   

  ```bash
   .
   ├── datasets
   │      ├── Kvasir-SEG
   │      │     ├── images
   │      │     │     └── *.jpg
   │      │     └── masks
   │      │           └── *.jpg
   │      ├── ISIC18
   │      │      ├── X_tr_224x224.npy 
   │      │      └── Y_tr_224x224.npy 
   │      │
   │      └── BUSI_sorted 
   │             ├── images
   │             │    ├── benign *.png
   │             │    └── malign *.png  
   │             └── masks 
   │                  ├── benign *.png
   │                  └── malign *.png  
   └── 
   ```         

## Supplementary notebooks
We have included a notebook with that serves as a supplement for the experiments around the Excitation PCA Threshold (EPT) in the `notebooks` folder.

## References 
[1] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Roy, Abhijit Guha, Nassir Navab, and Christian Wachinger. "Recalibrating fully convolutional networks with spatial and channel “squeeze and excitation” blocks." IEEE transactions on medical imaging 38.2 (2018): 540-549.

[3] Lee, HyunJae, Hyo-Eun Kim, and Hyeonseob Nam. "Srm: A style-based recalibration module for convolutional neural networks." Proceedings of the IEEE/CVF International conference on computer vision. 2019.

[4] Qin, Zequn, et al. "Fcanet: Frequency channel attention networks." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[5] Codella, Noel, et al. "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic)." arXiv preprint arXiv:1902.03368 (2019).

[6] Jha, Debesh, et al. "Kvasir-seg: A segmented polyp dataset." MultiMedia modeling: 26th international conference, MMM 2020, Daejeon, South Korea, January 5–8, 2020, proceedings, part II 26. Springer International Publishing, 2020.

[7] Al-Dhabyani, Walid, et al. "Dataset of breast ultrasound images." Data in brief 28 (2020): 104863.

[8] Azad, Reza, et al. "Medical image segmentation review: The success of u-net." arXiv preprint arXiv:2211.14830 (2022).

