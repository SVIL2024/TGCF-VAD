## Paper Title
TGCF-VAD: A Unified Framework via Triple-branch Gated Cross-modal Fusion for Video Anomaly Detection

## File structure
```
.
|-- README.md
|-- ckpt  # save checkpoints
|   |-- my_best
|   |   |-- 
|-- config.py
|-- dataset.py
|-- list  # ground truth and list files for training and testing
|    |-- gt-ped2.npy
|    |-- gt-sh2.npy
|    |-- gt-violence.npy
|    |-- ped2-i3d-test.list
|    |-- ped2-i3d.list
|    |-- shanghai-i3d-test-10crop.list
|    |-- shanghai-i3d-train-10crop.list
|    |-- violence-i3d-test.list
|    `-- violence-i3d.list
|-- main.py  # main file, train and test
|-- model.py  
|-- option.py  
|-- requirement.txt
|-- results
|-- save  # save features
|   |-- Shanghai
|   |   |-- SH_ten_crop_i3d_v2
|   |   `-- sent_emb_n
|   |-- UCSDped2
|   |   |-- ped2_ten_crop_i3d
|   |   `-- sent_emb_n
|   `-- Violence
|       |-- Violence_five_crop_i3d_v1
|       `-- sent_emb_n
|-- train.py
`-- utils.py
```

## Text features
Download from [LINK](https://pan.baidu.com/s/1tnCclFEFOI3CDY60w4tL6Q)提取码: tc6r (the file structure is the same as the tree map shown above) and put under `/save/Crime/snet_emb_n/` folder or generate the text features using this [repo](https://github.com/coranholmes/SwinBERT)

## Visual features
1. You can download from [here](https://pan.baidu.com/s/1lLLo-ca250ycIJjF0NUYRw)提取码: jw9t or generate the visual features using this [repo](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet).
2. For UCSDped2 dataset, put the generated/downloaded features under `./save/UCSDped2/UCF_ten_crop_i3d_v1` folder. Other datasets follow the same structure.
3. For UCSDped2 dataset, change the path of visual features in `./list/ped2-i3d-test.list` and `list/ped2-i3d.list`. Other datasets follow the same structure.

4. list [LINK](https://pan.baidu.com/s/1xD4wDPja6e-hbkymPEarWQ)提取码: jr4x

## Install requirements
Run `pip install -r requirement.txt` to install the requirements.

## Run visdom
**!!!VERY IMPORTANT!!!**

Open a separate terminal and run `visdom` after installing the requirements before running the following commands.

# Training + Testing
Meanings of the arguments can be seen in `option.py`. To train the best model presented in the paper, use the following settings:

ShanghaiTech dataset
```bash
python main.py --dataset shanghai_v2 --feature-group both --fusion crossattn_3glu --aggregate_text --extra_loss --batch-size 8
```
XD-Violence dataset
```bash
python main.py --dataset violence --feature-group both --fusion crossattn_3glu --aggregate_text --extra_loss --feature-size 1024 --batch-size 16 --alpha 0.00007
```
UCSD-Ped2 dataset
```bash
python main.py --dataset ped2 --feature-group both --fusion crossattn_3glu --aggregate_text --max-epoch 15000 --extra_loss --batch-size 2
```