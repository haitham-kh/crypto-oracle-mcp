# Model Weights Directory

This folder is **gitignored** — model `.json` and `.pkl` files are not committed to the repo.

## How to populate this folder

### Option A — Train on Google Colab (recommended)
```python
!git clone https://github.com/haitham-kh/crypto-oracle-mcp.git
%cd crypto-oracle-mcp
!python colab_v6_train.py
```
After training, download `MyDrive/crypto_oracle/models_v6/` from Google Drive and copy all files here.

### Option B — Train locally
```bash
python train_v6.py --processed-dir "E:\training data for quant\processed_features"
```

## Expected files after training

```
data/
├── v6_full_clf_long_h60.json
├── v6_full_clf_short_h60.json
├── v6_full_clf_long_h720.json
├── v6_full_clf_short_h720.json
├── v6_full_calib_long_h60.pkl
├── v6_full_calib_short_h60.pkl
├── v6_full_calib_long_h720.pkl
├── v6_full_calib_short_h720.pkl
├── v6_micro_clf_long_h60.json
├── v6_micro_clf_short_h60.json
├── v6_micro_clf_long_h720.json
├── v6_micro_clf_short_h720.json
├── v6_micro_calib_long_h60.pkl
├── v6_micro_calib_short_h60.pkl
├── v6_micro_calib_long_h720.pkl
├── v6_micro_calib_short_h720.pkl
├── v6_full_reg_h60.json
├── v6_full_reg_h720.json
└── v6_meta.json
```
