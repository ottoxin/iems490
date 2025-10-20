# Datasets

This directory contains sample datasets and data processing scripts for IEMS490.

## Dataset Categories

### Text Corpora
- Small text samples for quick experiments
- Preprocessed datasets for assignments
- Domain-specific corpora

### Benchmark Datasets
- Task-specific evaluation sets
- Standard NLP benchmarks
- Custom evaluation data

### Training Data
- Pre-training data samples
- Fine-tuning datasets
- Synthetic data examples

## Data Organization

```
data/
├── raw/                # Raw, unprocessed data
├── processed/          # Cleaned and preprocessed data
├── samples/            # Small samples for testing
└── scripts/            # Data processing scripts
```

## Using the Datasets

### Loading Data

```python
import pandas as pd

# Example: Load a CSV dataset
df = pd.read_csv('data/processed/example_dataset.csv')

# Example: Load from Hugging Face
from datasets import load_dataset
dataset = load_dataset('dataset_name')
```

### Preprocessing

Common preprocessing steps:
1. Text cleaning (removing special characters, etc.)
2. Tokenization
3. Normalization
4. Train/validation/test split
5. Creating data loaders

## Data Processing Scripts

Scripts in the `scripts/` subdirectory help with:
- Downloading datasets
- Cleaning and preprocessing
- Format conversion
- Data augmentation
- Statistics and analysis

## Important Notes

### Large Files
- Large datasets are not stored in Git
- Use `.gitignore` to exclude large files
- Download instructions provided for public datasets
- Use symbolic links or mount points for very large datasets

### Data Privacy
- Do not commit sensitive or private data
- Follow data usage agreements
- Ensure proper licensing for all datasets
- Anonymize data when necessary

### Reproducibility
- Document data sources and versions
- Include dataset statistics
- Provide data processing code
- Set random seeds for splits

## Common Datasets for LLMs

### Pre-training
- **The Pile**: Diverse text corpus
- **C4**: Cleaned Common Crawl
- **Wikipedia**: Encyclopedia text
- **BookCorpus**: Long-form text

### Fine-tuning
- **GLUE**: General language understanding
- **SuperGLUE**: More challenging tasks
- **SQuAD**: Question answering
- **RACE**: Reading comprehension

### Domain-Specific
- **PubMed**: Biomedical literature
- **arXiv**: Scientific papers
- **GitHub**: Code repositories
- **Legal**: Legal documents

## Dataset Best Practices

1. **Document thoroughly**: Describe source, license, and usage
2. **Version control**: Track dataset versions
3. **Quality checks**: Validate data quality
4. **Split properly**: Maintain separate train/val/test sets
5. **Sample datasets**: Provide small samples for quick testing

## Downloading Datasets

Most datasets should be downloaded separately due to size:

```bash
# Example: Download using wget
wget <dataset_url> -O data/raw/dataset.tar.gz
tar -xzvf data/raw/dataset.tar.gz -C data/raw/

# Example: Using Hugging Face CLI
huggingface-cli download dataset_name --repo-type dataset
```

## Data Statistics

When adding a dataset, include:
- Number of examples
- Train/validation/test split sizes
- Text length statistics
- Label distributions
- Any relevant metadata

## Contributing

When contributing new datasets:
1. Add documentation to this README
2. Include processing scripts
3. Provide sample data
4. Document licensing
5. Add to `.gitignore` if large

## Need Help?

For questions about datasets:
- Check the course discussion forum
- Consult the resources directory
- Ask during office hours
- Review Hugging Face Datasets documentation
