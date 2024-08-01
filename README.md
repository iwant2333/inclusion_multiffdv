
## Baseline Models

### Requirements

The main versions are,
- Python >= 3.7, < 3.11
- PyTorch >= 1.13
- torchvision >= 0.14
- pytorch_lightning == 1.7.*

Run the following command to install the required packages.

```bash
pip install -r requirements.txt
```

### Dataset

Download the dataset
```
data
  multiFFDV
    phase1
      - trainset
      - valset
      trainset_label.txt
      valset_label.txt
```

#### Video Examples
<video width='224' height='224' controls>
  <source src='./data/multiFFDV/phase1/trainset/6e14a72bccaaf5567209fc356ce04ab9.mp4' type='video/mp4'>
  <source src='./data/multiFFDV/phase1/trainset/30baacb39d9cc4abbdeb8c8c3c694450.mp4' type='video/mp4'>
  <source src='./data/multiFFDV/phase1/trainset/c4a9a62676e2031d4fe0b18764d16139.mp4' type='video/mp4'>
</video>

### Training

Train model

```bash
python train_binary_inclusion.py \
  --config ./config/batfd_plus_avdf1m.toml \
  --data_root ./data/multiFFDV/phase1/ \
  --batch_size 4 --num_workers 8 --gpus 1 --precision 32
```

The checkpoint will be saved in `./batfdp/avdf/` directory, and the tensorboard log will be saved in `lighntning_logs` directory.


### Evaluation

Please run the following command to evaluate the model with the checkpoint saved in `./batfdp/avdf/` directory.

```bash
python valid_binary.py \
  --config ./config/batfd_plus_avdf1m.toml \
  --data_root ./data/multiFFDV/phase1 \
  --checkpoint ./batfdp/avdf/Jul17_13-54-34/checkpoint_1_0_loss0.28512129187583923.pth/
```

The result will be save in prediction.txt

