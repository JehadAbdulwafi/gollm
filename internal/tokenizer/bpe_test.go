package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestBPE_SaveLoad(t *testing.T) {
	tests := []struct {
		name     string
		corpus   string
		maxVocab int
	}{
		{
			name:     "basic vocabulary",
			corpus:   "test corpus for serialization",
			maxVocab: 10,
		},
		{
			name:     "empty vocabulary",
			corpus:   "",
			maxVocab: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bpe1 := NewBPE(tt.maxVocab)
			bpe1.Train(tt.corpus)

			tmpFile := filepath.Join(t.TempDir(), "vocab.json")
			t.Cleanup(func() { os.Remove(tmpFile) })

			if err := bpe1.SaveVocab(tmpFile); err != nil {
				t.Fatalf("SaveVocab() error = %v", err)
			}

			bpe2 := NewBPE(tt.maxVocab)
			if err := bpe2.LoadVocab(tmpFile); err != nil {
				t.Fatalf("LoadVocab() error = %v", err)
			}

			if !reflect.DeepEqual(bpe1.vocab.id2word, bpe2.vocab.id2word) {
				t.Errorf("token to ID mapping mismatch")
			}
			if !reflect.DeepEqual(bpe1.vocab.word2id, bpe2.vocab.word2id) {
				t.Errorf("ID to token mapping mismatch")
			}
		})
	}
}

func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestBPE_Encode(t *testing.T) {
	tests := []struct {
		name  string
		text  string
		want  []int
		setup func(*BytePairEncoder)
	}{
		{
			name: "simple encoding",
			text: "what are you doing",
			want: []int{135, 157, 69, 1880},
			setup: func(bpe *BytePairEncoder) {
				if err := bpe.LoadVocab("../../data/vocab/vocab.json"); err != nil {
					t.Fatalf("Failed to load vocab: %v", err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bpe := NewBPE(0)
			tt.setup(bpe)

			t.Logf("Vocabulary: %v", bpe.vocab.word2id)
			t.Logf("Merges: %v", bpe.merges)
			got := bpe.Encode(tt.text)
			if !slicesEqual(got, tt.want) {
				t.Errorf("Encode() = %v, want %v", got, tt.want)
			}
		})
	}
}
