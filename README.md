# exploring-through-rcgvf

This repository supplements the paper ["Exploring through Random Curiosity with General Value Functions"](https://arxiv.org/abs/2211.10282). The code for the MiniGrid experiments is available in this repository. 

Code for the Diabolical Lock experiments is available [here](https://github.com/Aditya-Ramesh-10/diabolical-lock-experiments)

The implementation focuses on clarity and flexibility rather than computational efficiency.


## Instructions

Run an individual experiment with RC-GVF from the root directory:
```bash
python3 -m scripts.train_rcgvf

# Or with RND
python3 -m scripts.train_rnd

# Or with NovelD
python3 -m scripts.train_noveld

# Or with AGAC
python3 -m scripts.train_agac
```

The settings for environment and algorithm can be modified in the `config_defaults` dictionary in each training file, or through a [Weights & Biases (wandb)](https://docs.wandb.ai/) sweep. Details of baselines and the selected environment specific hyperparameters are available in Appendix C of our [paper](https://arxiv.org/abs/2211.10282).


In case you don't want to utilise Weights & Biases:
```bash
export WANDB_MODE=disabled
```


## Dependencies

- gymnasium
- [torch-ac](https://github.com/lcswillems/torch-ac)
- numpy
- torch
- wandb

## Acknowledgements

The code follows the structure used in [rl_starter_files](https://github.com/lcswillems/rl-starter-files).

