#!/bin/bash

export MKL_THREADING_LAYER=sequential

# main real anomaly experiment
for datatype in "thyroid" "breastw" "U2R" "lympho" "musk" "arrhythmia"; do
    # python expt_real.py $datatype na na sup
    for modelscoretype in "gmm energy" "vae recerr" "vae negelbo" "dagmm energy"; do
        for attrtype in "ASH"; do #"marg" "ACE" "SFE" "IG" "KSH" "wKSH" "comp" "ASH"; do
            python expt_real.py $datatype $modelscoretype $attrtype
        done
    done
done
