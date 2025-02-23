package configs

type ModelConfig struct {
	// Model architecture
	VocabSize   int `json:"vocab_size"`
	ContextSize int `json:"context_size"`
	EmbedDim    int `json:"embed_dim"`
	NumHeads    int `json:"num_heads"`
	NumLayers   int `json:"num_layers"`

	// Training configuration
	LearningRate float32 `json:"learning_rate"`
	BatchSize    int     `json:"batch_size"`
	MaxEpochs    int     `json:"max_epochs"`

	// Generation settings
	DefaultTemperature float32 `json:"default_temperature"`
	MaxTokens          int     `json:"max_tokens"`

	// Paths
	ModelPath     string `json:"model_path"`
	VocabPath     string `json:"vocab_path"`
	CheckpointDir string `json:"checkpoint_dir"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *ModelConfig {
	return &ModelConfig{
		VocabSize:          50257, // GPT-2 default
		ContextSize:        256,   // Increased context window
		EmbedDim:           384,   // Increased embedding dimension
		NumHeads:           6,     // Increased attention heads
		NumLayers:          6,     // Increased layers
		LearningRate:       1e-4,
		BatchSize:          32,
		MaxEpochs:          10,
		DefaultTemperature: 0.7,
		MaxTokens:          100,
		ModelPath:          "models/gollm.pt",
		VocabPath:          "data/vocab/vocab.json",
		CheckpointDir:      "checkpoints",
	}
}

// small
// 		VocabSize:   len(tok.Encoder),
// 		ContextSize: 256, // Increased context window
// 		EmbedDim:    384, // Increased embedding dimension
// 		NumHeads:    6,   // Increased attention heads
// 		NumLayers:   6,   // Increased layers
// 		LearningRate:       1e-4,
// 		BatchSize:          32,
// 		MaxEpochs:          10,
// 		DefaultTemperature: 0.7,
// 		MaxTokens:          100,
// 		ModelPath:          "models/gollm.pt",
// 		VocabPath:          "data/vocab/vocab.json",
// 		CheckpointDir:      "checkpoints",

// default
// 		VocabSize:          50257, // GPT-2 default
// 		ContextSize:        1024,
// 		EmbedDim:           768,
// 		NumHeads:           12,
// 		NumLayers:          12,
// 		LearningRate:       1e-4,
// 		BatchSize:          32,
// 		MaxEpochs:          10,
// 		DefaultTemperature: 0.7,
// 		MaxTokens:          100,
// 		ModelPath:          "models/gollm.pt",
// 		VocabPath:          "data/vocab/vocab.json",
// 		CheckpointDir:      "checkpoints",
