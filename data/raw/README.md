---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 56097337.0
    num_examples: 1440
  - name: test
    num_bytes: 14126775.0
    num_examples: 360
  download_size: 70180083
  dataset_size: 70224112.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
