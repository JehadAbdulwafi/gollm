package commands

import (
	"fmt"
	"gollm/internal/tokenizer"
	"log"

	"github.com/spf13/cobra"
)

var encodeCmd = &cobra.Command{
	Use:   "encode",
	Short: "Encode text using the trained tokenizer",
	Run: func(cmd *cobra.Command, args []string) {
		vocabPath, _ := cmd.Flags().GetString("vocab")
		text, _ := cmd.Flags().GetString("text")
		encodeText(vocabPath, text)
	},
}

func init() {
	encodeCmd.Flags().StringP("vocab", "v", "", "Path to the vocabulary file")
	encodeCmd.Flags().StringP("text", "t", "", "Text to encode")
	encodeCmd.MarkFlagRequired("vocab")
	encodeCmd.MarkFlagRequired("text")
}

func encodeText(vocabPath string, text string) {
	bpe := tokenizer.NewBPE(0)
	if err := bpe.LoadVocab(vocabPath); err != nil {
		log.Fatalf("Error loading vocabulary: %v\nDid you train the tokenizer first?", err)
	}

	ids := bpe.Encode(text)

	fmt.Println("Encoded tokens:")
	fmt.Println(ids)

	fmt.Println("\nToken breakdown:")
	for _, id := range ids {
		token := bpe.Vocab().GetWord(id)
		fmt.Printf("%q -> %d\n", token, id)
	}
}
