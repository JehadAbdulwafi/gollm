# GoLLM

A lightweight implementation of a transformer-based language model in Go.

## Features

- Transformer-based language model implementation (GPT-2 architecture)
- BPE tokenizer with vocabulary management
- Text generation with temperature control
- Model checkpointing and state persistence
- Configurable model architecture (small and default configurations)

## Usage

### 1. Train Tokenizer
First, train the BPE tokenizer on your corpus:
```bash
gollm train --corpus path/to/corpus.txt --vocab-size 50257
```

### 2. Pretrain Model
Pretrain the model on your corpus using either the default or small configuration:
```bash
# Using default config
gollm pretrain --corpus path/to/corpus.txt

# Using custom config
gollm pretrain --corpus path/to/corpus.txt --config path/to/config.json
```

### 3. Generate Text
Generate text using the trained model:
```bash
gollm generate --model path/to/model.pt --vocab path/to/vocab.json --prompt "Once upon a time"
```

### 4. Encode Text
Encode text using the trained tokenizer:
```bash
gollm encode --vocab path/to/vocab.json --text "Once upon a time"
```

## Model Configurations

### Default Configuration
Similar to GPT-2 small:
```json
{
  "vocab_size": 50257,
  "context_size": 1024,
  "embed_dim": 768,
  "num_heads": 12,
  "num_layers": 12,
  "learning_rate": 1e-4,
  "batch_size": 32,
  "max_epochs": 10
}
```

## Model File Format
The model weights are saved in `.pt` files with:
- Magic number identifier ("GoLM")
- Version number for compatibility
- JSON-encoded model state including:
  - Model configuration
  - Token and position embeddings
  - Transformer layer weights
  - Layer normalization parameters
  - Language model head weights

## TODO:

  - [ ] Add basic backpropagation and optimizer
  - [ ] Implement learning rate scheduling
  - [ ] Add training state checkpointing
  - [ ] Basic CUDA support for GPU acceleration
  - [ ] Add model quantization for smaller footprint
  - [ ] Implement flash attention
  - [ ] Add support for larger context windows
  - [ ] Optimize tensor operations
  - [ ] Add batched inference

## License

MIT License
