# Code Examples

This directory contains code examples and demonstrations for IEMS490.

## Structure

```
code/
├── basics/              # Fundamental concepts
├── transformers/        # Transformer implementation
├── training/            # Training examples
├── fine_tuning/         # Fine-tuning demonstrations
├── inference/           # Inference and generation
├── evaluation/          # Evaluation scripts
└── utilities/           # Helper functions and tools
```

## Code Examples by Topic

### Basics
- Tokenization examples
- Embedding layers
- Positional encoding
- Basic attention mechanisms

### Transformers
- Self-attention implementation
- Multi-head attention
- Transformer encoder
- Transformer decoder
- Full transformer model

### Training
- Data loading and preprocessing
- Training loops
- Optimization and learning rate schedules
- Mixed precision training
- Distributed training basics

### Fine-tuning
- Full model fine-tuning
- LoRA implementation
- Adapter methods
- Prompt tuning

### Inference
- Text generation strategies
- Beam search
- Top-k and top-p sampling
- Temperature scaling
- Batched inference

### Evaluation
- Perplexity calculation
- Task-specific metrics
- Model comparison utilities
- Benchmark evaluation

## Running the Examples

1. Ensure you have installed all requirements:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Navigate to the specific example directory

3. Follow the instructions in each example's notebook or script

## Jupyter Notebooks

Many examples are provided as Jupyter notebooks for interactive exploration:
- Clear explanations
- Step-by-step code
- Visualization of results
- Exercises and challenges

## Python Scripts

Some examples are provided as standalone Python scripts for:
- Production-ready code
- Command-line execution
- Performance benchmarking
- Automated testing

## Best Practices

The code examples follow these best practices:
- Clear documentation and comments
- Type hints where applicable
- Error handling
- Modular design
- Reproducible results (with seed setting)

## Contributing

If you have suggestions for new examples or improvements to existing ones, please:
1. Fork the repository
2. Create a feature branch
3. Add your example with documentation
4. Submit a pull request

## Notes

- Examples are designed for educational purposes
- Some examples may require GPU for reasonable performance
- Start with the basics directory if you're new to the topic
- Check resource requirements before running large examples
