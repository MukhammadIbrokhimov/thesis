# Kaggle API Setup Instructions

## Step 1: Get Your Kaggle API Token

1. Go to https://www.kaggle.com/
2. Sign in (or create an account if you don't have one)
3. Click on your profile picture (top right) → **Settings**
4. Scroll down to **API** section
5. Click **"Create New Token"**
6. This will download a file called `kaggle.json`

## Step 2: Install the Token

Run these commands in your terminal:

```bash
# Create .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json there
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (required for security)
chmod 600 ~/.kaggle/kaggle.json
```

## Step 3: Test the Setup

```bash
# Activate virtual environment
cd experiments
source venv/bin/activate

# Test Kaggle CLI
kaggle datasets list
```

If you see a list of datasets, you're all set!

## Step 4: Download the Datasets

Once configured, you can download the datasets:

```bash
# Dataset 1: Disease Symptom Prediction (~5,000 records)
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p data/raw/dataset1/ --unzip

# Dataset 2: Diseases and Symptoms (~246,000 records)
kaggle datasets download -d dhivyeshrk/diseases-and-symptoms-dataset -p data/raw/dataset2/ --unzip
```

## Troubleshooting

**Error: "Could not find kaggle.json"**
- Make sure `kaggle.json` is in `~/.kaggle/` directory
- Check file permissions: `ls -la ~/.kaggle/kaggle.json` should show `-rw-------`

**Error: "401 Unauthorized"**
- Your token may have expired. Go back to Kaggle settings and create a new token

**Error: "403 Forbidden"**
- You may need to accept the dataset's terms and conditions on the Kaggle website first
- Visit the dataset pages and click "Download" once (even if you cancel) to accept terms
