package tokenizer

import (
	"testing"
)

func TestBPE_Training(t *testing.T) {
	corpus := "low lower newest newest"
	bpe := NewBPE(25)
	bpe.Train(corpus)

	if len(bpe.merges) < 2 {
		t.Fatalf("Expected at least 2 merges, got %d", len(bpe.merges))
	}

	// Expected merges based on the corpus
	expectedMerges := []string{"w e", "e we", "ewe s", "ewes t", "l o", "n ewest", "newest </w>", "lo w", "lo we", "lowe r", "low </w>", "lower </w>"}

	if len(bpe.merges) < 2 {
		t.Errorf("Expected at least 2 merges, got %d", len(bpe.merges))
	}

	for i, expected := range expectedMerges {
		if i >= len(bpe.merges) {
			t.Errorf("Missing merge %q at position %d", expected, i)
			continue
		}
		if bpe.merges[i] != expected {
			t.Errorf("Merge %d:\nExpected %q\nGot      %q", i, expected, bpe.merges[i])
		}
	}
	// length of vocab + </w> should be 21
	if bpe.vocab.Len() != 21 {
		t.Errorf("Expected vocab size 21, got %d", bpe.vocab.Len())
	}

}

func TestBPE_SaveLoad(t *testing.T) {
	bpe1 := NewBPE(10)
	bpe1.Train("test corpus for serialization")

	tmpFile := t.TempDir() + "/vocab.json"
	if err := bpe1.SaveVocab(tmpFile); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	bpe2 := NewBPE(0)
	if err := bpe2.LoadVocab(tmpFile); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if len(bpe1.merges) != len(bpe2.merges) {
		t.Errorf("Merges mismatch: %d vs %d", len(bpe1.merges), len(bpe2.merges))
	}

	if bpe1.vocab.Len() != bpe2.vocab.Len() {
		t.Errorf("Vocab size mismatch: %d vs %d", bpe1.vocab.Len(), bpe2.vocab.Len())
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

