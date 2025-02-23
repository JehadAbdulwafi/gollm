package tokenizer

import (
	"encoding/json"
	"fmt"
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
	fmt.Println("Preprocessing corpus...")
	words := b.preprocessCorpus(corpus)
	fmt.Printf("Found %d words\n", len(words))

	fmt.Println("Initializing vocabulary...")
	b.initializeVocab(words)
	fmt.Printf("Initial vocabulary size: %d\n", b.vocab.Len())

	iteration := 0
	for b.vocab.Len() < b.maxVocab {
		iteration++
		fmt.Printf("Training iteration %d (vocab size: %d/%d)\n", iteration, b.vocab.Len(), b.maxVocab)

		pairs := b.getPairs(words)
		if len(pairs) == 0 {
			fmt.Println("No more pairs to merge")
			break
		}

		bestPair := b.selectBestPair(pairs)
		fmt.Printf("Selected pair: %q (frequency: %d)\n", bestPair, pairs[bestPair])

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
	words := strings.Fields(corpus)
	processed := make([]string, 0, len(words))

	b.vocab.Add("<s>")   // Start of text
	b.vocab.Add("</s>")  // End of text
	b.vocab.Add("<unk>") // Unknown token
	b.vocab.Add("<pad>") // Padding token
	b.vocab.Add(" ")     // Space token

	for _, word := range words {
		if len(word) == 0 {
			continue
		}

		chars := []string{" "}
		for _, r := range word {
			chars = append(chars, string(r))
		}
		chars = append(chars, "</w>")

		processed = append(processed, strings.Join(chars, " "))
	}

	return processed
}

func (b *BytePairEncoder) initializeVocab(words []string) {
	for _, word := range words {
		chars := strings.Split(word, " ")
		for _, char := range chars {
			b.vocab.Add(char)
		}
	}
}

func (b *BytePairEncoder) getPairs(words []string) map[string]int {
	pairs := make(map[string]int)
	for _, word := range words {
		chars := strings.Split(word, " ")
		for i := 0; i < len(chars)-1; i++ {
			pair := chars[i] + " " + chars[i+1]
			pairs[pair]++
		}
	}
	return pairs
}

func (b *BytePairEncoder) selectBestPair(pairs map[string]int) string {
	type pair struct {
		p string
		c int
	}

	pairList := make([]pair, 0, len(pairs))
	for p, c := range pairs {
		pairList = append(pairList, pair{p, c})
	}

	sort.Slice(pairList, func(i, j int) bool {
		if pairList[i].c != pairList[j].c {
			return pairList[i].c > pairList[j].c
		}
		return pairList[i].p < pairList[j].p
	})

	if len(pairList) > 0 {
		return pairList[0].p
	}
	return ""
}

func (b *BytePairEncoder) mergePair(words []string, bestPair string) []string {
	parts := strings.Split(bestPair, " ")
	if len(parts) != 2 {
		return words
	}

	merged := strings.ReplaceAll(bestPair, " ", "")
	result := make([]string, len(words))

	for i, word := range words {
		chars := strings.Split(word, " ")
		var newChars []string

		for j := 0; j < len(chars); j++ {
			if j < len(chars)-1 && chars[j] == parts[0] && chars[j+1] == parts[1] {
				newChars = append(newChars, merged)
				j++
			} else {
				newChars = append(newChars, chars[j])
			}
		}

		result[i] = strings.Join(newChars, " ")
	}

	return result
}

func (b *BytePairEncoder) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	words := strings.Fields(text)
	var tokens []int

	for i, word := range words {
		if len(word) == 0 {
			continue
		}

		if i > 0 {
			if spaceID := b.vocab.GetID(" "); spaceID >= 0 {
				tokens = append(tokens, spaceID)
			}
		}

		chars := make([]string, 0, len(word))
		for _, c := range word {
			chars = append(chars, string(c))
		}
		chars = append(chars, "</w>")
		current := strings.Join(chars, " ")

		for _, merge := range b.merges {
			parts := strings.Split(merge, " ")
			if len(parts) != 2 {
				continue
			}
			current = strings.ReplaceAll(current, merge, strings.ReplaceAll(merge, " ", ""))
		}

		subwords := strings.Split(current, " ")
		for _, subword := range subwords {
			if id := b.vocab.GetID(subword); id >= 0 {
				tokens = append(tokens, id)
			} else if unkID := b.vocab.GetID("<unk>"); unkID >= 0 {
				tokens = append(tokens, unkID)
			}
		}
	}

	return tokens
}
