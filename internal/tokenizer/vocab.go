package tokenizer

import (
	"sort"
)

type Vocab struct {
	word2id map[string]int
	id2word map[int]string
	counts  map[string]int
}

func NewVocab() *Vocab {
	return &Vocab{
		word2id: make(map[string]int),
		id2word: make(map[int]string),
		counts:  make(map[string]int),
	}
}

func (v *Vocab) Add(word string) {
	if _, exists := v.word2id[word]; !exists {
		id := len(v.word2id)
		v.word2id[word] = id
		v.id2word[id] = word
	}
	v.counts[word]++
}

func (v *Vocab) Len() int {
	return len(v.word2id)
}

func (v *Vocab) GetID(word string) int {
	if id, exists := v.word2id[word]; exists {
		return id
	}
	return -1
}

func (v *Vocab) GetWord(id int) string {
	return v.id2word[id]
}

func (v *Vocab) SortedByCount() []string {
	type kv struct {
		word  string
		count int
	}

	var ss []kv
	for k, v := range v.counts {
		ss = append(ss, kv{k, v})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].count > ss[j].count
	})

	result := make([]string, len(ss))
	for i, kv := range ss {
		result[i] = kv.word
	}
	return result
}
