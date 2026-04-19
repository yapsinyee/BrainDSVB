# HPC Sync Workflow

This repo already has a Git remote:

```bash
git remote -v
```

Current upstream:

```bash
origin  https://github.com/yapsinyee/BrainDSVB.git
```

## 1. Push your local branch from your laptop/workstation

From the repo root:

```bash
cd "/Users/syyap/Programming/Monash x Cambridge/ST-BrainXDL"
git status
git add README.md main.py model.py requirements.txt step1_compute_ldw.py step2_prepare_data.py train.py .gitignore HPC_SYNC.md
git commit -m "Add COBRE fMRI dynamic preprocessing and HPC sync notes"
git push origin mps-support
```

If you want to use a fresh branch for the HPC run instead:

```bash
git checkout -b cobre-fmri-hpc
git push -u origin cobre-fmri-hpc
```

## 2. Clone or update on M3

On the allocated M3 node:

```bash
git clone https://github.com/yapsinyee/BrainDSVB.git
cd BrainDSVB
git checkout mps-support
```

If the repo is already cloned on M3:

```bash
cd BrainDSVB
git fetch origin
git checkout mps-support
git pull --ff-only origin mps-support
```

## 3. Create or activate the environment on M3

If you already have a conda env:

```bash
conda activate braindsvb
pip install -r requirements.txt
```

If not:

```bash
conda create -n braindsvb python=3.10
conda activate braindsvb
pip install -r requirements.txt
```

## 4. Put large data on M3 separately

Git should only carry code and lightweight text assets. Do not sync generated outputs through Git.

Keep these local or copy them to M3 via `rsync`/project storage instead:

- `data/cobre_fmri/`
- `data/ldw_data/`
- `data/folds_data/`
- `saved_models/`
- `logs/`

Example copy command to M3 after you have a project path:

```bash
rsync -avP "/local/path/to/data/cobre_fmri/" username@host:/path/on/m3/BrainDSVB/data/cobre_fmri/
```

## 5. Run preprocessing and training on M3

Dynamic COBRE fMRI preprocessing:

```bash
python step1_compute_ldw.py \
  --dataset cobre_fmri \
  --connectivity-mode dynamic \
  --window-size 20 \
  --shift 10 \
  --min-windows 14
```

Prepare folds:

```bash
python step2_prepare_data.py --dataset cobre_fmri --connectivity-mode dynamic
```

Train one fold:

```bash
python main.py --dataset cobre_fmri --connectivity-mode dynamic --outer-loop 1 --inner-loop 1
```

Train the 5 outer folds with `inner-loop 1`:

```bash
for outer in 1 2 3 4 5; do
  python main.py --dataset cobre_fmri --connectivity-mode dynamic --outer-loop "$outer" --inner-loop 1
done
```

## 6. What Git will ignore now

The repo ignore rules now exclude:

- `data/ldw_data/`
- `data/folds_data/`
- `saved_models/`
- `logs/`
- Python cache files

That keeps your HPC sync code-focused and avoids committing generated artifacts by accident.
