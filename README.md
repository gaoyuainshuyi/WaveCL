# WaveCL: Wavelet Calibration Learning for Referring Video Object Segmentation

### Environment Setup
 -  `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
 - `pip install h5py opencv-python protobuf av einops ruamel.yaml timm joblib pandas matplotlib cython scipy` 
 - `pip install transformers==4.24.0`
 - `pip install numpy==1.23.5`
 - `pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
 - build up MultiScaleDeformableAttention
    ```
    cd ./models/ops
    python setup.py build install
    ``` 

## Data Preparation
The Overall data preparation is set as followed.
We put rvosdata under the path `data/`.

```text
data
└── a2d_sentences/ 
    ├── Release/
    │   ├── videoset.csv  (videos metadata file)
    │   └── CLIPS320/
    │       └── *.mp4     (video files)
    └── text_annotations/
        ├── a2d_annotation.txt  (actual text annotations)
        ├── a2d_missed_videos.txt
        └── a2d_annotation_with_instances/ 
            └── */ (video folders)
                └── *.h5 (annotations files)
└── refer_youtube_vos/ 
    ├── train/
    │   ├── JPEGImages/
    │   │   └── */ (video folders)
    │   │       └── *.jpg (frame image files) 
    │   └── Annotations/
    │       └── */ (video folders)
    │           └── *.png (mask annotation files) 
    ├── valid/
    │   └── JPEGImages/
    │       └── */ (video folders)
    |           └── *.jpg (frame image files) 
    └── meta_expressions/
        ├── train/
        │   └── meta_expressions.json  (text annotations)
        └── valid/
            └── meta_expressions.json  (text annotations)
└── coco/
      ├── train2014/
      ├── refcoco/
        ├── instances_refcoco_train.json
        ├── instances_refcoco_val.json
      ├── refcoco+/
        ├── instances_refcoco+_train.json
        ├── instances_refcoco+_val.json
      ├── refcocog/
        ├── instances_refcocog_train.json
        ├── instances_refcocog_val.json
```

### Output Dir
We put all outputs under a dir. Specifically, We set `WCL` as the output dir, so please change it to `xxx/WCL`.

#### Training From scratch
We only use Video-Swin-T as backbone to train and eval the dataset.

- A2D
  Run the scripts `bash ./scripts/train_a2d.sh`.

  The key parameters are as follows and change the ./configs/a2d_sentences.yaml:
  | lr | backbone_lr | bs | GPU_num | Epoch | lr_drop |
  |:----:|:-------------:|:----:|:---------:|:-------:|:-------:|
  |5e-5| 5e-6        | 2  |  2      |  40   | 15(0.2) |

- Ref-Youtube-VOS 
  Run the "bash ./scripts/train_ytvos_scratch.sh.

 The main parameters are as follow:
  | lr |backbone_lr| bs | num_class | GPU_num | freeze_text_encoder| lr_drop| Epoch |
  |:----:|:----------:|:----:|:-----------:|:---------:|:--------:|:--------:|:------:|
  |1e-4| 1e-5     | 1  |  65       |   8     |     true           |  20(0.1)|  30   |

###  Pretrain and Finetune
We perform pretrain and finetune on A2d-Sentences and Ref-Youtube-VOS dataset using Video-Swin-Tiny. Following previous work, we first pretrain on RefCOCO dataset and then finetune.

- Pretrain

Run the scripts `bash ./scripts/pretrain.sh`. The followings are the key parameters for pretrain. When pretrain, please specify the corresponding backbone.

  | lr |backbone_lr|text_encoder_lr | bs | num_class | GPU_num | freeze_text_encoder| lr_drop| Epoch |
    |:----:|:----------:|:------:|:----:|:-----------:|:---------:|:--------:|:--------:|:------:|
    |1e-4| 1e-5  | 5e-6  | 8  |  1       |   8     |     False          |  15 20(0.1)|  30   |

- Ref-Youtube-VOS

  Run the scripts `bash ./scripts/finetune_ytb.sh`. We finetune the pretrained weight using the following key parameters:
  | lr |backbone_lr|text_encoder_lr | bs | num_class | GPU_num | freeze_text_encoder| lr_drop| Epoch |
    |:----:|:----------:|:------:|:----:|:-----------:|:---------:|:--------:|:--------:|:------:|
    |1e-4| 1e-5  | 5e-6 | 8  |  1       |   8     |     False         |  10(0.1)|  25   |

- A2D-Sentences

 Run the scripts `bash ./scripts/finetune_a2d.sh`. We finetune the pretrained weight on A2D-Sentences using the following key parameters:
  | lr |backbone_lr|text_encoder_lr | bs | num_class | GPU_num | freeze_text_encoder| lr_drop| Epoch |
    |:----:|:----------:|:------:|:----:|:-----------:|:---------:|:--------:|:--------:|:------:|
    |3e-5| 3e-6  | 1e-6 | 1  |  1       |   8     |     true         |  - |  20   |

### Evaluation

- A2D-Sentences

  Run the scripts `bash ./scripts/eval_a2d.sh`.

- JHMDB-Sentences 

Run the scripts `bash ./scripts/eval_jhd.sh`.

- Ref-Youtube-VOS

Run the scripts ` bash ./scripts/infer_refytb.sh`.

### Acknowledgement
Code in this repository is built upon several public repositories. Thanks for the wonderful work [Referformer](https://github.com/wjn922/ReferFormer),  [MTTR](https://github.com/mttr2021/MTTR) and [SOC](https://github.com/RobertLuo1/NeurIPS2023_SOC).









