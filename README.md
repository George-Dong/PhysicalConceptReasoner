# PhysicalConceptReasoner-Release

Pytorch implementation for paper *[Compositional Physical Reasoning of Objects and Events from Videos](https://arxiv.org/abs/2408.02687)*. More details and visualization results can be found at the [project page](https://physicalconceptreasoner.github.io/).

## Framework

<div align="center">
  <img src="assets/model_arch_cropped_00.png" width="100%">
</div>

**[Compositional Physical Reasoning of Objects and Events from Videos](https://arxiv.org/abs/2408.02687)**

[Zhenfang Chen](https://zfchenunique.github.io/),
[Shilong Dong](https://www.linkedin.com/in/shilong-dong/),
[Kexin Yi](https://scholar.google.fr/citations?view_op=list_works&hl=fr&hl=fr&user=ylUBBDwAAAAJ),
[Yunzhu Li](https://yunzhuli.github.io/),
[Mingyu Ding](https://dingmyu.github.io/),
[Antonio Torralba](https://web.mit.edu/torralba/www/),
[Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html),
[Chuang Gan](http://people.csail.mit.edu/ganchuang/)

## Installation

- Prerequisites

  - Python 3
  - PyTorch 1.0 or higher, with NVIDIA CUDA Support
  - Other required python packages specified by `requirements.txt`.
- Install [Jacinle](https://github.com/vacancy/Jacinle): Clone the package, and add the bin path to your global `PATH` environment variable:

  ```
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<path_to_jacinle>/bin:$PATH
  ```
- Clone this repository:

  ```
  git clone https://github.com/zfchenUnique/DCL-Release.git --recursive
  ```
- Create a conda environment for NS-CL, and install the requirements. This includes the required python packages from both Jacinle NS-CL. Most of the required packages have been included in the built-in `anaconda` package.

## Dataset preparation

- Download videos, video annotation,  questions and answers, and object proposals accordingly from the [official website](https://physicalconceptreasoner.github.io/#)
- Transform videos into ".png" frames with ffmpeg.

## Step-by-step Training on ComPhy Dataset

- Step 1: download the [proposals](http://clevrer.csail.mit.edu/#) from the region proposal network and extract object trajectories for train and val set by

```
   bash scripts/script_gen_tubes.sh
```

- Step 2: train a concept learner with descriptive and explanatory questions for static concepts (i.e. color, shape and material)

```
   bash scripts/comphy_train_pcr_stage1 <GPU_ID> <DATA_DIR>
```

- Step 3: extract static attributes & refine object trajectories
  extract static attributes

```
   bash scripts/script_extract_attribute.sh
```

    refine object trajectories

```
   bash scripts/script_gen_tubes_refine.sh
```

- Step 4: train a pcr for stage2 learning

```
    bash script/script_comphy_train_pcr_stage2.sh <GPU_ID> <DATA_DIR> <STAGE1_MODEL_DIR>
```

- Step 5: train a pcr for stage3 learning

```
    bash script/script_comphy_train_pcr_stage3.sh <GPU_ID> <DATA_DIR> <STAGE2_MODEL_DIR>

```

## Generalization to Real-World Scenario Dataset

- Step 1: Download [training videos](https://drive.google.com/file/d/18wBkxmee1rEUU0Cmdx9N8eUoXhbKie8f/view), [validation videos](https://drive.google.com/file/d/1nL2_zWUdrOLZQXJHkzCqXzQn7phgzKNo/view) and [related questions](https://drive.google.com/file/d/1did2gIUhx8zRHnwxl8lkwjlkcaD-b90p/view) from google drive.
- Step 2: Finetune a pretrained PCR model on the Real-World Scenario dataset.

```
    bash script/script_real_world_dataset_finetune.sh <GPU_ID> <DATA_DIR> <STAGE2_MODEL_DIR>

```

## Citation

If you find this repo useful in your research, please consider citing:

```
@misc{chen2024compositionalphysicalreasoningobjects,
      title={Compositional Physical Reasoning of Objects and Events from Videos}, 
      author={Zhenfang Chen and Shilong Dong and Kexin Yi and Yunzhu Li and Mingyu Ding and Antonio Torralba and Joshua B. Tenenbaum and Chuang Gan},
      year={2024},
      eprint={2408.02687},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02687}, 
}
```
