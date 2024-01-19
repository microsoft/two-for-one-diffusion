# Sammpling commands
This file contains the sampling commands used for each protein to generate the i.i.d. samples and Langevin dynamics results reported in the [paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00702):


## Langevin dynamics sampling commands

#### CHIGNOLIN
```bash
python sample.py 
    --model_path saved_models/chignolin
    --gen_mode langevin
    --noise_level 20
    --parallel_sim 100
    --n_timesteps 6000000 
    --save_interval 500
    --kb consistent  
    --dt 2e-3 
```

### TRP-CAGE
```bash
python sample.py 
    --model_path saved_models/trp_cage
    --gen_mode langevin
    --noise_level 15
    --parallel_sim 100
    --n_timesteps 6000000 
    --save_interval 500
    --kb consistent  
    --dt 2e-3 
```

### BBA
```bash
python sample.py 
    --model_path saved_models/bba
    --gen_mode langevin
    --noise_level 5
    --parallel_sim 100
    --n_timesteps 6000000 
    --save_interval 500
    --kb consistent  
    --dt 2e-3 
```

### VILLIN
```bash
python sample.py 
    --model_path saved_models/villin
    --gen_mode langevin
    --noise_level 5
    --parallel_sim 100
    --n_timesteps 6000000 
    --save_interval 500
    --kb consistent  
    --dt 2e-3 
```

### PROTEIN_G
```bash
python sample.py 
    --model_path saved_models/protein_g
    --gen_mode langevin
    --noise_level 5
    --parallel_sim 100
    --n_timesteps 6000000 
    --save_interval 500
    --kb consistent  
    --dt 2e-3 
```

### ALANINE DIPEPTIDE
The different noise levels used for Alanine Dipeptide can be checked in the paper, Table S4.

```bash
python sample.py 
    --model_path saved_models/alanine/fold1
    --gen_mode langevin
    --noise_level 8
    --parallel_sim 100
    --n_timesteps 1000000 
    --save_interval 250
    --kb consistent  
    --dt 2e-3 
```


## I.i.d. Sampling commands

#### CHIGNOLIN
```bash
python sample.py 
    --model_path saved_models/chignolin
    --gen_mode iid
    --num_samples_eval 374320
    --batch_size_gen 256
```

### TRP-CAGE
```bash
python sample.py 
    --model_path saved_models/trp_cage
    --gen_mode iid
    --num_samples_eval 730800
    --batch_size_gen 256
```

### BBA
```bash
python sample.py 
    --model_path saved_models/bba
    --gen_mode iid
    --num_samples_eval 780181
    --batch_size_gen 256
```

### VILLIN
```bash
python sample.py 
    --model_path saved_models/villin
    --gen_mode iid
    --num_samples_eval 439535
    --batch_size_gen 256
```

### PROTEIN G
```bash
python sample.py 
    --model_path saved_models/protein_g
    --gen_mode iid
    --num_samples_eval 1294476
    --batch_size_gen 128
```


### ALANINE DIPEPTIDE
```bash
python sample.py 
    --model_path saved_models/alanine/fold1
    --gen_mode iid
    --num_samples_eval 400000
    --batch_size_gen 128
```