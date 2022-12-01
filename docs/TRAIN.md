# Training

We provide pre-set config files for each experiment setting in `configs` folder.
Make sure to configure the dataset paths in config files and run the commands from the main directory `ViFi-CLIP/`.
Below we provide training instructions for ViFi-CLIP and its variants.

Instructions:
- [Zero-shot setting](#Zero-shot)
- [Base-to-novel generalization setting](#Base-to-novel-generalization)
- [Few-shot setting](#Few-shot)
- [Fully-supervised setting](#Fully-supervised)
- [Vanilla CLIP zero-shot evaluation](#Vanilla-ZS-CLIP)


### Zero-shot

We train all models on Kinetics-400 with 32 frames for 10 epochs and then evaluate directly on downstream datasets (HMDB-51, UCF-101 and K-600). All zero-shot config files are present at `configs/zero_shot/*.yaml`.

To train ViFi-CLIP model on Kinetics-400, run the following command. 

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/zero_shot/train/k400/16_16_vifi_clip.yaml --output /PATH/TO/OUTPUT 
```

Note: If you want to finetune on either image or text encoder of CLIP, please set the variable `USE` in the config file to "image" or "text" respectively.
For example, to train CLIP with only text fine-tuning, run the following command:

```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/zero_shot/train/k400/16_16_vifi_clip.yaml --output /PATH/TO/OUTPUT --opts TRAINER.ViFi_CLIP.USE "text"
```

#### Testing zero-shot
Use the configs at `configs/zero_shot/eval` to evaluate the trained models on downstream datasets. 

For example, to evaluate ViFi-CLIP trained model on Kinetics-600 first split, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/zero_shot/eval/k600/16_32_K600_ZS_split1.yaml --output /PATH/TO/OUTPUT --only_test \
 --resume /PATH/TO/TRAINED/VIFI-CLIP-CKPT
```


### Base-to-novel-generalization
The default training settings are configured in config files at `configs/base2novel`. Below we provide instructions to train 
and evaluate ViFi-CLIP model on Kinetics-400.


```bash
# seed=1
# trains and evaluates on base classes, this will save weights named "ckpt_epoch_10.pth"
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/base2novel/finetuning_base2novel/k400/16_32_vifi_clip_s1.yaml --output /PATH/TO/OUTPUT 
# evaluates on novel classes, use model weights which are trained on base classes
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/base2novel/finetuning_base2novel/k400/16_32_vifi_clip_novel_eval.yaml --output /PATH/TO/OUTPUT --only_test \
 --resume /PATH/TO/TRAINED/VIFI-CLIP-CKPT
```
To train using other variants of CLIP, modify the config parameter `TRAINER.ViFi_CLIP.USE` accordingly.


#### VL prompting approach
Config files for VL prompting approach are provided at `configs/base2novel/prompting_base2novel`. Use the pretrained ViFi-CLIP model to initialize training when using VL prompting approach. 

Below we provide instructions to train VL prompting method in base-to-novel setting on HMDB-51.

```bash
# seed=1
# use pretrained model (on K-400) for training
# trains and evaluates on base classes, 
# this will save weights named "ckpt_epoch_10.pth", 
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/base2novel/prompting_base2novel/hmdb/16_32_prompting_s1.yaml --output /PATH/TO/OUTPUT \ 
--resume /PATH/TO/TRAINED-K400/VIFI-CLIP-CKPT
# evaluates on novel classes, use model weights which are trained on base classes
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/base2novel/prompting_base2novel/hmdb/16_32_prompting_novel_eval.yaml --output /PATH/TO/OUTPUT --only_test \
 --resume /PATH/TO/TRAINED/VIFI-CLIP-CKPT
```
This trains only the vision and language prompts on the downstream task while rest of CLIP model is kept frozen.

Similarly, using the corresponding config files, models can be trained on other datasets including HMDB-51, UCF-101 and SSv2.


### Few-shot

Use the config files at `configs/few_shot` to train models in few-shot setting.
Below we provide instructions to train ViFi-CLIP on HMDB-51 in few-shot manner for K=2.

```bash
# K=2
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_2_shot.yaml --output /PATH/TO/OUTPUT 
```
To train using other variants of CLIP, modify the config parameter `TRAINER.ViFi_CLIP.USE` accordingly.


#### VL prompting approach
We provide config files for VL prompting approach at `configs/few_shot/prompting_few_shot`.
Use the pretrained ViFi-CLIP model to initialize training when using VL prompting approach. 

Below we provide instructions to train VL prompting method on HMDB-51 in few-shot manner for K=2.

```bash
# K=2
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/few_shot/prompting_few_shot/hmdb51/16_32_prompting_2_shot.yaml --output /PATH/TO/OUTPUT \
--resume /PATH/TO/TRAINED-K400/VIFI-CLIP-CKPT
```
This trains only the vision and language prompts on the downstream task in few-shot manner while rest of CLIP model is kept frozen.


### Fully-supervised
For fully-supervised experiments, we provide config files at `configs/fully_supervised/k400`. 

For example, to train ViFi-CLIP (tunes both image and text encoder) on Kinetics-400, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 \ 
main.py -cfg configs/fully_supervised/k400/16_16_vifi_clip.yaml --output /PATH/TO/OUTPUT 
```
To train using other variants of CLIP, modify the config parameter `TRAINER.ViFi_CLIP.USE` accordingly.


### Vanilla-ZS-CLIP
Here we provide evaluation instructions to evaluate Vanilla CLIP (without any video training). All experimental settings are supported.

All config files can be used directly. Just turn ON the `ZS_EVAL` flag while evaluating and it will use vanilla CLIP for evaluation.
For example, to evaluate ZS vanilla CLIP on HMDB-51 split-1, run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/base2novel/prompting_base2novel/hmdb/16_32_prompting_novel_eval.yaml --output /PATH/TO/OUTPUT --only_test \
 --resume "" --opts TRAINER.ViFi_CLIP.ZS_EVAL "True"
```
This will evaluate the vanilla ZS CLIP on the given dataset.





