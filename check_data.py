import os, glob

root = r'c:\Users\skuna\cryptogame\crypto-oracle-mcp'

# Check 1: Colab training_data parquets
colab_dir = os.path.join(root, 'data', 'processed')
colab_files = glob.glob(os.path.join(colab_dir, '*_training_data.parquet'))
print(f'=== Colab processed parquets (data/processed/) ===')
print(f'  Found: {len(colab_files)} files')
for f in sorted(colab_files)[:5]:
    sz = os.path.getsize(f)/1e6
    print(f'  {os.path.basename(f)}  ({sz:.1f} MB)')

# Check 2: E-drive raw monthly parquets
edrive_dirs = [
    r'E:\training data for quant\processed_features',
    r'E:\training data for quant',
    r'E:\crypto_oracle_data\processed',
]
print()
print('=== E-drive raw monthly parquets ===')
for d in edrive_dirs:
    if os.path.exists(d):
        files  = glob.glob(os.path.join(d, '**', '*_1m_features_*.parquet'), recursive=True)
        files2 = glob.glob(os.path.join(d, '**', '*_training_data.parquet'), recursive=True)
        all_f  = files + files2
        total_mb = sum(os.path.getsize(f) for f in all_f) / 1e6
        print(f'  DIR: {d}')
        print(f'    monthly parquets   : {len(files)}')
        print(f'    training_data parq : {len(files2)}')
        print(f'    total size         : {total_mb:.0f} MB')
        coins = sorted(set(os.path.basename(f).split('_1m_')[0] for f in files))
        if coins:
            print(f'    coins (monthly)    : {coins}')
        coins2 = sorted(set(os.path.basename(f).split('_training_data')[0] for f in files2))
        if coins2:
            print(f'    coins (training)   : {coins2}')
    else:
        print(f'  NOT FOUND: {d}')

# Check 3: existing V5 model files
print()
print('=== Existing model files (data/) ===')
model_files = (glob.glob(os.path.join(root, 'data', '*.json'))
             + glob.glob(os.path.join(root, 'data', '*.pkl')))
for f in sorted(model_files):
    sz = os.path.getsize(f)/1e6
    print(f'  {os.path.basename(f)}  ({sz:.1f} MB)')
if not model_files:
    print('  (none found)')

# Check 4: challenge_data
print()
cdata = os.path.join(root, 'challenge_data')
if os.path.exists(cdata):
    cf = glob.glob(os.path.join(cdata, '*.parquet'))
    print(f'=== challenge_data/ === ({len(cf)} parquets)')
    for f in sorted(cf)[:8]:
        print(f'  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.1f} MB)')
else:
    print('=== challenge_data/ === NOT FOUND')
