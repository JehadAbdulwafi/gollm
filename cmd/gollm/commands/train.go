package commands

import (
	"fmt"
	"gollm/internal/tokenizer"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train the tokenizer on a corpus",
	Run: func(cmd *cobra.Command, args []string) {
		corpusPath, _ := cmd.Flags().GetString("corpus")
		vocabSize, _ := cmd.Flags().GetString("vocab-size")
		trainTokenizer(corpusPath, vocabSize)
	},
}

func init() {
	trainCmd.Flags().StringP("corpus", "i", "", "Path to the corpus file")
	trainCmd.Flags().StringP("vocab-size", "v", "", "Vocabulary size")
	trainCmd.MarkFlagRequired("corpus")
	trainCmd.MarkFlagRequired("vocab-size")
}

func trainTokenizer(corpusPath string, vocabSize string) {
	maxVocab, err := strconv.Atoi(vocabSize)
	if err != nil {
		log.Fatalf("Invalid vocab size: %v", err)
	}

	content, err := os.ReadFile(corpusPath)
	if err != nil {
		log.Fatalf("Error reading corpus: %v", err)
	}

	bpe := tokenizer.NewBPE(maxVocab)
	bpe.Train(string(content))

	vocabFile := fmt.Sprintf("%s-vocab-%d.json", strings.TrimSuffix(corpusPath, ".txt"), maxVocab)
	if err := bpe.SaveVocab(vocabFile); err != nil {
		log.Fatalf("Error saving vocabulary: %v", err)
	}

	fmt.Printf("Tokenizer trained successfully!\nVocabulary saved to: %s\n", vocabFile)
}
