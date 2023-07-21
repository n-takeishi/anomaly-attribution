# A Characteristic Function for Shapley-Value-Based Attribution of Anomaly Scores

Codes for the method presented in the following paper:

Naoya Takeishi and Yoshinobu Kawahara, A Characteristic Function for Shapley-Value-Based Attribution of Anomaly Scores, *Transactions on Machine Learning Research*, 2023.

https://openreview.net/forum?id=eLX5XrajXh

## Procedures

(1) Train detectors and run detections in advance

`python pre_train.py` (train detector models)

`python pre_detect.py` (run anomaly detection)

(2) Synthetic anomaly experiment

`bash expt_synth.sh` (execute anomaly attribution on synthetic anomalies)

`python stats_synth.py` (output statistics based on the attribution results)

`python table_synth.py` (output tables based on the statistics)

(3) Real anomaly experiment

`bash expt_real.sh` (execute anomaly attribution on real anomalies)

`python stats_real.py`

`python table_real.py`

## Notes

Each of the datasets is originally from:

- U2R: https://github.com/InitRoot/NSLKDD-Dataset
- Thyroid: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
- Musk: http://odds.cs.stonybrook.edu/musk-dataset/
- WBC: http://odds.cs.stonybrook.edu/wbc/
- BreastW: http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/
- Arrhythmia: http://odds.cs.stonybrook.edu/arrhythmia-dataset/

## Author

Naoya Takeishi

https://ntake.jp/
