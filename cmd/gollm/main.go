package main

import (
	"fmt"
	"gollm/internal/tokenizer"
	"log"
	"os"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: gollm [train|encode]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "train":
		if len(os.Args) < 4 {
			log.Fatal("Usage: gollm train <corpus.txt> <vocab_size>")
		}
		trainTokenizer(os.Args[2], os.Args[3])

	case "encode":
		if len(os.Args) < 4 {
			log.Fatal("Usage: gollm encode <vocab.json> <text>")
		}
		encodeText(os.Args[2], os.Args[3])

	default:
		log.Fatal("Unknown command")
	}
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
