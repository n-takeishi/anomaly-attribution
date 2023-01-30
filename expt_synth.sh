#!/bin/bash

export MKL_THREADING_LAYER=sequential

# main synthetic anomaly experiment
for datatype in "thyroid" "breastw" "U2R" "lympho" "musk" "arrhythmia"; do
    for num_anom in 1 2 3; do
        for modelscoretype in "gmm energy" "vae recerr" "vae negelbo" "dagmm energy"; do
            for attrtype in "marg" "ACE" "SFE" "IG" "KSH" "wKSH" "comp" "ASH"; do
                python expt_synth.py $datatype $num_anom $modelscoretype $attrtype
            done
        done
    done
done

# hyperparameter sensitivity
num_anom=1
modelscoretype="gmm energy"
attrtype="ASH"
for datatype in "thyroid" "breastw" "U2R" "lympho" "musk" "arrhythmia"; do
    for gamma in 0.001 0.1 0.01 1.0 10.0; do
        python expt_synth.py $datatype $num_anom $modelscoretype $attrtype $gamma
    done
done

# examine relaxation's effect
num_anom=1
attrtype="ASHexact"
for datatype in "breastw"; do
    for modelscoretype in "gmm energy" "vae recerr" "vae negelbo" "dagmm energy"; do
        python expt_synth.py $datatype $num_anom $modelscoretype $attrtype
    done
done
