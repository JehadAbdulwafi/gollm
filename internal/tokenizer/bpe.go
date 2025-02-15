package tokenizer

import (
	"encoding/json"
	"os"
	"sort"
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
		chars := make([]string, 0, len(word))
		for _, c := range word {
			chars = append(chars, string(c))
		}
		chars = append(chars, "</w>")
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
	)

	// First pass: find max count
	for _, count := range pairs {
		if count > maxCount {
			maxCount = count
		}
	}

	// Second pass: find all pairs with max count
	var candidates []string
	for pair, count := range pairs {
		if count == maxCount {
			candidates = append(candidates, pair)
		}
	}

	// Prioritize pairs without word-end markers first
	sort.Slice(candidates, func(i, j int) bool {
		hasI := strings.Contains(candidates[i], "</w>")
		hasJ := strings.Contains(candidates[j], "</w>")
		if hasI == hasJ {
			return candidates[i] < candidates[j]
		}
		return !hasI
	})

	if len(candidates) > 0 {
		return candidates[0]
	}
	return ""
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

	// Join all words and split into symbols
	allSymbols := strings.Fields(strings.Join(processed, " "))

	// Apply merges in reverse order (most recent first)
	for i := len(b.merges) - 1; i >= 0; i-- {
		merge := b.merges[i]
		parts := strings.Split(merge, " ")
		if len(parts) != 2 {
			continue
		}

		var newSymbols []string
		j := 0
		for j < len(allSymbols) {
			if j < len(allSymbols)-1 &&
				allSymbols[j] == parts[0] &&
				allSymbols[j+1] == parts[1] {
				newSymbols = append(newSymbols, parts[0]+parts[1])
				j += 2
			} else {
				newSymbols = append(newSymbols, allSymbols[j])
				j++
			}
		}
		allSymbols = newSymbols
	}

	// Convert to IDs with unknown handling
	unkID := b.vocab.GetID("<UNK>")
	ids := make([]int, 0, len(allSymbols))
	for _, s := range allSymbols {
		if id, exists := b.vocab.word2id[s]; exists {
			ids = append(ids, id)
		} else if unkID != -1 {
			ids = append(ids, unkID)
		}
	}
	return ids
}
