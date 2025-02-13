package tokenizer

import (
	"encoding/json"
	"log"
	"os"
	"strings"
)

type BytePairEncoder struct {
	merges   []string
	vocab    *Vocab
	maxVocab int
}

func NewBPE(maxVocab int) *BytePairEncoder {
	return &BytePairEncoder{
		merges:   []string{},
		vocab:    NewVocab(),
		maxVocab: maxVocab,
	}
}

func (b *BytePairEncoder) Train(corpus string) {
	words := b.preprocessCorpus(corpus)
	b.initializeVocab(words)

	for b.vocab.Len() < b.maxVocab {
		pairs := b.getPairs(words)
		if len(pairs) == 0 {
			break
		}

		bestPair := b.selectBestPair(pairs)
		words = b.mergePair(words, bestPair)
		b.merges = append(b.merges, bestPair)
		b.vocab.Add(strings.ReplaceAll(bestPair, " ", ""))
	}
}

func (b *BytePairEncoder) Vocab() *Vocab {
	return b.vocab
}

func (b *BytePairEncoder) SaveVocab(path string) error {
	data := struct {
		Merges []string       `json:"merges"`
		Vocab  map[string]int `json:"vocab"`
	}{
		Merges: b.merges,
		Vocab:  b.vocab.word2id,
	}

	file, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, file, 0644)
}

func (b *BytePairEncoder) LoadVocab(path string) error {
	file, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var data struct {
		Merges []string       `json:"merges"`
		Vocab  map[string]int `json:"vocab"`
	}

	if err := json.Unmarshal(file, &data); err != nil {
		return err
	}

	b.merges = data.Merges
	b.vocab = NewVocab()
	for word, id := range data.Vocab {
		b.vocab.word2id[word] = id
		b.vocab.id2word[id] = word
	}

	return nil
}

func (b *BytePairEncoder) preprocessCorpus(corpus string) []string {
	preprocessor := NewPreprocessor(true)
	cleanText := preprocessor.Clean(corpus)
	normalizedText := preprocessor.NormalizeWhitespace(cleanText)

	words := strings.Fields(normalizedText)
	processed := make([]string, 0, len(words))

	for _, word := range words {
		chars := make([]string, 0, len(word)+1)
		for _, c := range word {
			chars = append(chars, string(c))
		}
		processed = append(processed, strings.Join(chars, " "))
	}
	return processed
}

func (b *BytePairEncoder) initializeVocab(words []string) {
	for _, word := range words {
		for _, symbol := range strings.Fields(word) {
			b.vocab.Add(symbol)
		}
	}
}

func (b *BytePairEncoder) getPairs(words []string) map[string]int {
	pairs := make(map[string]int)
	for _, word := range words {
		symbols := strings.Fields(word)
		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			pairs[pair]++
		}
	}
	return pairs
}

func (b *BytePairEncoder) selectBestPair(pairs map[string]int) string {
	var (
		maxCount int
		bestPair string
	)

	for pair, count := range pairs {
		if count > maxCount || (count == maxCount && pair < bestPair) {
			bestPair = pair
			maxCount = count
		}
	}
	return bestPair
}

func (b *BytePairEncoder) mergePair(words []string, bestPair string) []string {
	merged := strings.ReplaceAll(bestPair, " ", "")
	newWords := make([]string, 0, len(words))

	for _, word := range words {
		symbols := strings.Fields(word)
		var newSymbols []string
		i := 0

		for i < len(symbols) {
			if i < len(symbols)-1 &&
				symbols[i]+" "+symbols[i+1] == bestPair {
				newSymbols = append(newSymbols, merged)
				i += 2
			} else {
				newSymbols = append(newSymbols, symbols[i])
				i++
			}
		}
		newWords = append(newWords, strings.Join(newSymbols, " "))
	}

	return newWords
}

func (b *BytePairEncoder) Encode(text string) []int {
	processed := b.preprocessCorpus(text)
	if len(processed) == 0 {
		return nil
	}

	symbols := strings.Fields(strings.Join(processed, " "))
	log.Printf("symbols: %v", symbols)

	// Apply learned merges IN ORDER
	for _, merge := range b.merges {
		parts := strings.Split(merge, " ")
		if len(parts) != 2 {
			continue
		}

		var newSymbols []string
		i := 0
		for i < len(symbols) {
			if i < len(symbols)-1 &&
				symbols[i] == parts[0] &&
				symbols[i+1] == parts[1] {
				newSymbols = append(newSymbols, parts[0]+parts[1])
				i += 2
			} else {
				newSymbols = append(newSymbols, symbols[i])
				i++
			}
		}
		symbols = newSymbols
	}

	unkID := b.vocab.GetID("<UNK>")
	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, exists := b.vocab.word2id[s]; exists {
			ids = append(ids, id)
		} else if unkID != -1 {
			ids = append(ids, unkID)
		}
	}
	return ids
}
