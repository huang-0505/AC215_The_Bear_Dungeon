# D&D Narrator Dataset Creator

This tool loads the CRD3 (Critical Role Dungeons & Dragons Dialogue) dataset from Hugging Face and prepares it for fine-tuning Gemini models to act as a D&D narrator and NPC character.

## Dataset

**CRD3 Dataset**: Contains transcripts from Critical Role, a popular D&D actual play web series. The dataset includes rich narrative descriptions, character dialogue, and gameplay interactions perfect for training a D&D narrator model.

- Source: `microsoft/crd3` on Hugging Face
- Total Samples: ~52,796 dialogue chunks
  - Train: 38,969 samples
  - Validation: 6,327 samples
  - Test: 7,500 samples
- Contains: D&D gameplay transcripts with narration and dialogue
- Use case: Fine-tuning Gemini to be an immersive D&D narrator and NPC roleplayer

## Usage

### 1. Build the Docker Container

First, make sure you're in the dataset-creator directory and build the container:

```bash
cd src/finetuning/llm-finetuning/dataset-creator
bash docker-shell.sh
```

Or if you already built it and just want to run:

```bash
bash docker-run.sh
```

### 2. Prepare the Dataset

#### Option A: Process ALL Data (Recommended for Production)

Load and process all train, validation, and test splits from CRD3:

```bash
# Inside the container - process all ~52k samples
python cli.py --prepare --all
```

This will:
1. Download the CRD3 dataset from Hugging Face
2. Process **all 38,969 training samples**
3. Process **all 6,327 validation samples**
4. Process **all 7,500 test samples**
5. Format data for Gemini fine-tuning (JSONL format)
6. Save files to the `data/` folder (see Output Files below)

#### Option B: Process Limited Samples (For Testing)

For quick testing or development, process a limited number of samples:

```bash
# Inside the container - process only 1000 samples
python cli.py --prepare --max-samples 1000
```

Options:
- `--prepare`: Load and prepare the CRD3 dataset
- `--all`: Process ALL data from all three splits (train, validation, test)
- `--max-samples`: Number of samples to use from train split only (default: 1000, ignored if `--all` is used)

### 3. Upload to GCS Bucket

Upload the prepared dataset to Google Cloud Storage:

```bash
python cli.py --upload
```

This uploads all prepared files to: `gs://{YOUR_BUCKET}/dnd-narrator-finetune-dataset/`

### 4. Combined Workflow

You can run both steps together:

```bash
# Process all data and upload
python cli.py --prepare --all --upload

# Or for testing with limited samples
python cli.py --prepare --max-samples 5000 --upload
```

## Environment Variables

Make sure these are set (usually in your `.env` file):

- `GCP_PROJECT`: Your Google Cloud project ID
- `GCS_BUCKET_NAME`: Your GCS bucket name for storing datasets

## Output Files

### When using `--all` flag (All Data):

**JSONL Files** (for Gemini fine-tuning):
- `train.jsonl` - All 38,969 training samples
- `validation.jsonl` - 256 validation samples (Gemini's max for validation)
- `validation_full.jsonl` - All 6,327 validation samples (for your analysis)
- `test.jsonl` - All 7,500 test samples (for final evaluation)

**CSV Files** (human-readable):
- `train.csv` - All training samples in text format
- `validation.csv` - All validation samples in text format
- `test.csv` - All test samples in text format
- `dnd-instruct-dataset.csv` - Combined dataset with all samples

### When using `--max-samples` (Limited Data):

- `train.jsonl` - ~90% of processed samples for training
- `validation.jsonl` - ~10% of processed samples for validation (max 256)
- `train.csv`, `validation.csv` - Text format versions
- `dnd-instruct-dataset.csv` - Full processed dataset

### File Structure:

```
data/
├── dnd-instruct-dataset.csv    # All processed samples combined
├── train.jsonl                 # Training data for Gemini
├── train.csv                   # Training data (readable)
├── validation.jsonl            # Validation data for Gemini (max 256)
├── validation.csv              # Validation data (readable)
├── validation_full.jsonl       # Full validation set (all samples)
├── test.jsonl                  # Test data (if --all used)
└── test.csv                    # Test data (readable, if --all used)
```

## Data Format

The prepared data follows Gemini's fine-tuning format:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Continue the D&D story and narrate what happens next:"}]
    },
    {
      "role": "model",
      "parts": [{"text": "The tavern door creaks open..."}]
    }
  ]
}
```

## Understanding the Splits

- **Train** (38,969 samples): Used to train/fine-tune the model
- **Validation** (6,327 samples): Used during training to monitor performance and prevent overfitting
  - `validation.jsonl` is limited to 256 samples (Gemini's requirement)
  - `validation_full.jsonl` contains all validation samples for your own evaluation
- **Test** (7,500 samples): Used after training to evaluate the final model performance

## Fine-tuning the Model

After preparing and uploading the dataset, use the fine-tuning container to train your Gemini model:

```bash
cd ../gemini-finetuner
bash docker-shell.sh
# Follow fine-tuning instructions
```

For Gemini fine-tuning, you'll use:
- `train.jsonl` - Main training data
- `validation.jsonl` - Validation data (256 samples)

## Tips

- **Start small**: Use `--max-samples 1000` for testing the pipeline
- **Production**: Use `--all` to process the full dataset (~52k samples)
- **Dataset quality**: The CRD3 dataset is high-quality D&D narration from Critical Role
- **Training time**: More data = longer training time but better model quality
- **Validation**: The validation set helps prevent overfitting and improves model generalization
- **System instruction**: The model will be trained with a D&D narrator persona built into the system instruction
