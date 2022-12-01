

# Data Preparation

To prepare the datasets, we prefer to organize the dataset as follows:

-  **Standard Folder**. Put all videos in the `videos` folder, and prepare the annotation files as `train.txt` and `val.txt`. Please make sure the folder looks like this:
    ```Shell
    $ ls /PATH/TO/videos | head -n 2
    a.mp4
    b.mp4

    $ head -n 2 /PATH/TO/train.txt
    a.mp4 0
    b.mp4 2

    $ head -n 2 /PATH/TO/val.txt
    c.mp4 1
    d.mp4 2
    ```

Additionally, we provide prepared annotation text files containing train/test splits for all datasets in `datasets_splits` folder.


The instructions to prepare each dataset are detailed below. 

### HMDB51
- Download and unrar the HMDB51 dataset from the official website using this [link](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar).
- Arrange the videos of all classes in a single folder structure as shown above.
- Directly use the annotation files provided in `datasets_splits/hmdb_splits` folder.



### UCF101
- Download and unrar UCF101 dataset from the official website using this [link]()
- After extracting, the folder will be already in the required format (no need to arrange videos in a single folder).
- Directly use the annotation files provided in `datasets_splits/ucf_splits` folder.


### Kinetics
- For downloading and preparing K400 and K600 datasets, please follow the detailed steps provided [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README.md) or [CVDF](https://github.com/cvdfoundation/kinetics-dataset).
- All videos of training set should be present in `train` folder and videos for validation set should be present in `val` folder.
- Overall, the directory structure should look like:
```
k400/
|–– train/ # contains videos for training set
|   |–– ZzZfJghfIz8_000068_000078.mp4
|   |–– ZzzLmGNirTg_000001_000011.mp4
|   |–– ...
|–– val/ # contains videos for val set
|   |–– -kahgmRD-4g_000004_000014.mp4
|   |–– KaMfzeeYwWw_000042_000052.mp4
|   |–– ...
```
- Directly use the annotation files provided in `datasets_splits/k400_splits` folder


### Something Something v2 (SSv2)
- Download SSv2 dataset from the official website using this [link](https://developer.qualcomm.com/software/ai-datasets/something-something)
- Arrange the videos of all classes in a single folder structure similar to HMDB51 dataset.
- Directly use the annotation files provided in `datasets_splits/ssv2_splits`

NOTE: The dataloader directly loads videos in an online fashion using [decord](https://github.com/dmlc/decord).

ViFi-CLIP and its variants uses text-label embeddings for action classification. All labels files have been prepared and provided in `labels/` folder.
In order to create label file for your custom dataset, the format should look like this:
```Shell
$ head -n 5 labels/kinetics_400_labels.csv
id,name
0,abseiling
1,air drumming
2,answering questions
3,applauding
```
The `id` indicates the class id, while the `name` denotes the text description.


Acknowledgements: [XCLIP's repository](https://github.com/microsoft/VideoX/tree/master/X-CLIP).