package commands

import (
	"encoding/json"
	"fmt"
	"gollm/configs"
	"gollm/internal/model"
	"gollm/internal/tokenizer"
	"log"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

var pretrainCmd = &cobra.Command{
	Use:   "pretrain",
	Short: "Pretrain the model on a text corpus",
	Run: func(cmd *cobra.Command, args []string) {
		corpusPath, _ := cmd.Flags().GetString("corpus")
		configPath, _ := cmd.Flags().GetString("config")
		runPretrain(corpusPath, configPath)
	},
}

func init() {
	pretrainCmd.Flags().StringP("corpus", "i", "", "Path to the training corpus")
	pretrainCmd.Flags().StringP("config", "c", "", "Path to model config file (optional)")
	pretrainCmd.MarkFlagRequired("corpus")
	rootCmd.AddCommand(pretrainCmd)
}

func runPretrain(corpusPath, configPath string) {
	cfg := configs.DefaultConfig()
	if configPath != "" {
		configData, err := os.ReadFile(configPath)
		if err != nil {
			log.Fatalf("Error reading config file: %v", err)
		}
		if err := json.Unmarshal(configData, cfg); err != nil {
			log.Fatalf("Error parsing config file: %v", err)
		}
	}

	os.MkdirAll(filepath.Dir(cfg.ModelPath), 0755)
	os.MkdirAll(cfg.CheckpointDir, 0755)

	tok := tokenizer.New()
	if err := tok.Load(cfg.VocabPath); err != nil {
		log.Fatalf("Failed to load vocabulary: %v", err)
	}

	cfg.VocabSize = tok.VocabSize()

	gpt := model.NewGPT2(model.Config{
		VocabSize:   cfg.VocabSize,
		ContextSize: cfg.ContextSize,
		EmbedDim:    cfg.EmbedDim,
		NumHeads:    cfg.NumHeads,
		NumLayers:   cfg.NumLayers,
	})

	data, err := os.ReadFile(corpusPath)
	if err != nil {
		log.Fatalf("Error reading corpus: %v", err)
	}
	corpus := string(data)

	tokens := tok.Encode(corpus)

	fmt.Println("Starting pretraining...")
	batchSize := cfg.BatchSize
	numBatches := (len(tokens) - cfg.ContextSize) / batchSize

	for epoch := 0; epoch < cfg.MaxEpochs; epoch++ {
		totalLoss := float32(0)

		for batch := 0; batch < numBatches; batch++ {
			batchLoss := float32(0)

			for i := 0; i < batchSize; i++ {
				start := batch*batchSize + i
				if start+cfg.ContextSize >= len(tokens) {
					continue
				}

				sequence := tokens[start : start+cfg.ContextSize]
				target := tokens[start+1 : start+cfg.ContextSize+1]

				logits := gpt.Forward(sequence)

				loss := model.CrossEntropyLoss(logits, target)
				batchLoss += loss

				// Backward pass and optimization (to be implemented)
				// TODO: Implement backpropagation and parameter updates
			}

			batchLoss /= float32(batchSize)
			totalLoss += batchLoss

			if batch%100 == 0 {
				fmt.Printf("Epoch %d/%d, Batch %d/%d, Loss: %.4f\n",
					epoch+1, cfg.MaxEpochs, batch+1, numBatches, batchLoss)
			}
		}

		avgLoss := totalLoss / float32(numBatches)
		fmt.Printf("Epoch %d/%d complete, Average Loss: %.4f\n",
			epoch+1, cfg.MaxEpochs, avgLoss)

		checkpointPath := filepath.Join(cfg.CheckpointDir,
			fmt.Sprintf("checkpoint-epoch-%d.pt", epoch+1))
		if err := gpt.Save(checkpointPath); err != nil {
			log.Printf("Warning: Failed to save checkpoint: %v", err)
		}
	}

	if err := gpt.Save(cfg.ModelPath); err != nil {
		log.Fatalf("Error saving model: %v", err)
	}
	fmt.Printf("Training complete! Model saved to: %s\n", cfg.ModelPath)
}
