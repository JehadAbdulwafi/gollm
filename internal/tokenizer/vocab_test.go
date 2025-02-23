package tokenizer

import (
	"testing"
)

func TestVocab_AddGet(t *testing.T) {
	v := NewVocab()

	words := []string{"apple", "banana", "apple", "cherry"}
	for _, w := range words {
		v.Add(w)
	}

	assertIntEqual(t, v.counts["apple"], 2, "apple count")
	assertIntEqual(t, v.counts["banana"], 1, "banana count")

	assertIntEqual(t, v.GetID("apple"), 0, "apple ID")
	assertIntEqual(t, v.GetID("banana"), 1, "banana ID")
	assertIntEqual(t, v.GetID("missing"), -1, "missing ID")

	assertStringEqual(t, v.GetWord(0), "apple", "ID 0")
	assertStringEqual(t, v.GetWord(1), "banana", "ID 1")
}

// New helper functions for different types
func assertIntEqual(t *testing.T, actual, expected int, msg string) {
	t.Helper()
	if actual != expected {
		t.Errorf("%s: expected %d, got %d", msg, expected, actual)
	}
}

func assertStringEqual(t *testing.T, actual, expected string, msg string) {
	t.Helper()
	if actual != expected {
		t.Errorf("%s: expected %q, got %q", msg, expected, actual)
	}
}

func TestVocab_SortedByCount(t *testing.T) {
	v := NewVocab()
	v.Add("a") // 3 times
	v.Add("a")
	v.Add("a")
	v.Add("b") // 2 times
	v.Add("b")
	v.Add("c") // 1 time

	sorted := v.SortedByCount()
	expected := []string{"a", "b", "c"}

	if len(sorted) != len(expected) {
		t.Fatalf("Expected %d items, got %d", len(expected), len(sorted))
	}

	for i := range expected {
		if sorted[i] != expected[i] {
			t.Errorf("Position %d: expected %s, got %s", i, expected[i], sorted[i])
		}
	}
}

func TestVocab_UnknownHandling(t *testing.T) {
	v := NewVocab()
	v.Add("<UNK>")
	v.Add("test")

	assertEqual(t, v.GetID("test"), 1, "known word ID")

	assertEqual(t, v.GetID("missing"), -1, "unknown word ID")

	if id := v.GetID("<UNK>"); id != 0 {
		t.Errorf("Expected <UNK> ID 0, got %d", id)
	}
}

func assertEqual(t *testing.T, actual, expected int, msg string) {
	t.Helper()
	if actual != expected {
		t.Errorf("%s: expected %d, got %d", msg, expected, actual)
	}
}
