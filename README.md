# A Characteristic Function for Shapley-Value-Based Attribution of Anomaly Scores

## Procedures

(1) Train detectors and run detections in advance

`python pre_train.py`
`python pre_detect.py`

(2) Synthetic anomaly experiment

`bash expt_synth.sh`
`python stats_synth.py`
`python table_synth.py`

(3) Real anomaly experiment

`bash expt_real.sh`
`python stats_real.py`
`python table_real.py`
