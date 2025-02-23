package tokenizer

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strings"
)

type Tokenizer struct {
	Encoder map[string]int
	Decoder map[int]string
	Merges  []string
}

type vocabFile struct {
	Merges []string       `json:"merges"`
	Vocab  map[string]int `json:"vocab"`
}

func New() *Tokenizer {
	return &Tokenizer{
		Encoder: make(map[string]int),
		Decoder: make(map[int]string),
	}
}

func (t *Tokenizer) Train(text string, vocabSize int) error {
	t.Encoder = make(map[string]int)
	t.Decoder = make(map[int]string)
	t.Merges = nil

	t.Encoder["<unk>"] = 0
	t.Decoder[0] = "<unk>"
	t.Encoder["<pad>"] = 1
	t.Decoder[1] = "<pad>"
	t.Encoder["<cls>"] = 2
	t.Decoder[2] = "<cls>"
	t.Encoder["<sep>"] = 3
	t.Decoder[3] = "<sep>"
	t.Encoder["<mask>"] = 4
	t.Decoder[4] = "<mask>"

	pairs := make(map[string]int)
	tokens := strings.Split(text, "")

	for i := 0; i < len(tokens)-1; i++ {
		pair := tokens[i] + tokens[i+1]
		pairs[pair]++
	}

	for len(t.Encoder) < vocabSize {
		if len(pairs) == 0 {
			break
		}

		var bestPair string
		var maxCount int
		for pair, count := range pairs {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		t.Merges = append(t.Merges, bestPair)

		newToken := bestPair
		t.Encoder[newToken] = len(t.Encoder)
		t.Decoder[len(t.Decoder)] = newToken

		var newTokens []string
		for i := 0; i < len(tokens); i++ {
			if i < len(tokens)-1 && tokens[i]+tokens[i+1] == bestPair {
				newTokens = append(newTokens, newToken)
				i++
			} else {
				newTokens = append(newTokens, tokens[i])
			}
		}
		tokens = newTokens

		pairs = make(map[string]int)
		for i := 0; i < len(tokens)-1; i++ {
			pair := tokens[i] + tokens[i+1]
			pairs[pair]++
		}
	}

	return nil
}

func (t *Tokenizer) Load(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %v", err)
	}

	var vf vocabFile
	if err := json.Unmarshal(data, &vf); err != nil {
		return fmt.Errorf("failed to unmarshal vocab file: %v", err)
	}

	t.Encoder = vf.Vocab
	t.Merges = vf.Merges

	t.Decoder = make(map[int]string)
	for token, id := range t.Encoder {
		t.Decoder[id] = token
	}

	return nil
}

func (t *Tokenizer) Save(path string) error {
	vf := vocabFile{
		Merges: t.Merges,
		Vocab:  t.Encoder,
	}

	data, err := json.Marshal(vf)
	if err != nil {
		return fmt.Errorf("failed to marshal vocab file: %v", err)
	}

	return ioutil.WriteFile(path, data, 0644)
}

func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	tokens := strings.Split(text, "")
	for _, merge := range t.Merges {
		parts := strings.Split(merge, " ")
		if len(parts) != 2 {
			continue
		}

		var newTokens []string
		for i := 0; i < len(tokens); i++ {
			if i < len(tokens)-1 && tokens[i] == parts[0] && tokens[i+1] == parts[1] {
				newTokens = append(newTokens, merge)
				i++
			} else {
				newTokens = append(newTokens, tokens[i])
			}
		}
		tokens = newTokens
	}

	var result []int
	for _, token := range tokens {
		if id, ok := t.Encoder[token]; ok {
			result = append(result, id)
		} else {
			for _, b := range []byte(token) {
				result = append(result, int(b))
			}
		}
	}

	return result
}

func (t *Tokenizer) Decode(tokens []int) string {
	var result strings.Builder

	for _, token := range tokens {
		if text, ok := t.Decoder[token]; ok {
			switch text {
			case "<s>", "</s>", "<unk>", "<pad>":
				continue
			case " ":
				result.WriteString(" ")
			default:
				if strings.HasSuffix(text, "</w>") {
					text = strings.TrimSuffix(text, "</w>")
					result.WriteString(text + " ")
				} else {
					result.WriteString(text)
				}
			}
		}
	}

	return strings.TrimSpace(result.String())
}

// VocabSize returns the size of the vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.Encoder)
}
