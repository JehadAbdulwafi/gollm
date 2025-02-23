package commands

import (
	"fmt"
	"gollm/configs"
	"gollm/internal/model"
	"gollm/internal/tokenizer"
	"log"
	
	"github.com/spf13/cobra"
)

var generateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Generate text from a prompt",
	Long: `Generate text from a given prompt using the trained model.
Example: gollm generate "Once upon a time" --temperature 0.7`,
	Run: runGenerate,
}

func init() {
	rootCmd.AddCommand(generateCmd)
	
	generateCmd.Flags().StringP("model", "m", "", "path to model file")
	generateCmd.Flags().StringP("vocab", "v", "", "path to vocabulary file")
	generateCmd.Flags().StringP("prompt", "p", "", "text prompt to start generation")
	generateCmd.Flags().Float32P("temperature", "t", 0.7, "sampling temperature")
	generateCmd.Flags().IntP("max-tokens", "n", 100, "maximum number of tokens to generate")
	
	generateCmd.MarkFlagRequired("model")
	generateCmd.MarkFlagRequired("vocab")
	generateCmd.MarkFlagRequired("prompt")
}

// func generateText(vocabPath string, prompt string, temperature float32) {
// 	// Load tokenizer
// 	tok := tokenizer.New()
// 	if err := tok.Load(vocabPath); err != nil {
// 		log.Fatalf("Failed to load vocab: %v", err)
// 	}
//
// 	// Create model configuration
// 	cfg := model.Config{
// 		VocabSize:   len(tok.Encoder),
// 		ContextSize: 256, // Increased context window
// 		EmbedDim:    384, // Increased embedding dimension
// 		NumHeads:    6,   // Increased attention heads
// 		NumLayers:   6,   // Increased layers
// 	}
// 	gpt2 := model.NewGPT2(cfg)
//
// 	// Encode prompt
// 	tokens := tok.Encode(prompt)
//
// 	// Generate text with specified temperature
// 	maxTokens := len(prompt) + 100 // Generate 100 new tokens after prompt
// 	generated := gpt2.Generate(tokens, maxTokens, temperature)
//
// 	// Decode and print
// 	text := tok.Decode(generated)
// 	fmt.Println(text)
// }

func runGenerate(cmd *cobra.Command, args []string) {
	modelPath, _ := cmd.Flags().GetString("model")
	vocabPath, _ := cmd.Flags().GetString("vocab")
	prompt, _ := cmd.Flags().GetString("prompt")
	temperature, _ := cmd.Flags().GetFloat32("temperature")
	maxTokens, _ := cmd.Flags().GetInt("max-tokens")
	
	// Load tokenizer
	tok := tokenizer.New()
	if err := tok.Load(vocabPath); err != nil {
		log.Fatalf("Failed to load vocabulary: %v", err)
	}
	
	// Get default config and update with tokenizer vocab size
	cfg := configs.DefaultConfig()
	cfg.VocabSize = tok.VocabSize()
	
	// Initialize model with config
	m := model.NewGPT2(model.Config{
		VocabSize:   cfg.VocabSize,
		ContextSize: cfg.ContextSize,
		EmbedDim:    cfg.EmbedDim,
		NumHeads:    cfg.NumHeads,
		NumLayers:   cfg.NumLayers,
	})
	
	// Load model weights
	if err := m.Load(modelPath); err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	
	// Encode prompt
	tokens := tok.Encode(prompt)
	
	// Generate text
	generated := m.Generate(tokens, maxTokens, temperature)
	
	// Decode and print
	text := tok.Decode(generated)
	fmt.Println(text)
}
