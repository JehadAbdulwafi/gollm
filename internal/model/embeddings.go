package model

import (
	"math/rand"
	"sync"
)

type Embeddings struct {
	VocabSize     int
	EmbedDim      int
	ContextSize   int
	TokenEmbed    [][]float32
	PositionEmbed [][]float32
}

func NewEmbeddings(vocabSize, embedDim, contextSize int) *Embeddings {
	e := &Embeddings{
		VocabSize:     vocabSize,
		EmbedDim:      embedDim,
		ContextSize:   contextSize,
		TokenEmbed:    make([][]float32, vocabSize),
		PositionEmbed: make([][]float32, contextSize),
	}

	scale := float32(0.02)

	var wg sync.WaitGroup
	numWorkers := 4
	tokensPerWorker := (vocabSize + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Add(-1)
			startToken := workerID * tokensPerWorker
			endToken := (workerID + 1) * tokensPerWorker
			if endToken > vocabSize {
				endToken = vocabSize
			}

			for i := startToken; i < endToken; i++ {
				e.TokenEmbed[i] = make([]float32, embedDim)
				for j := range e.TokenEmbed[i] {
					e.TokenEmbed[i][j] = (rand.Float32()*2 - 1) * scale
				}
			}
		}(w)
	}
	wg.Wait()

	for i := range e.PositionEmbed {
		e.PositionEmbed[i] = make([]float32, embedDim)
		for j := range e.PositionEmbed[i] {
			e.PositionEmbed[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}

	return e
}

func (e *Embeddings) Lookup(tokens []int) [][]float32 {
	result := make([][]float32, len(tokens))

	var wg sync.WaitGroup
	wg.Add(len(tokens))

	for i, token := range tokens {
		go func(idx, tok int) {
			defer wg.Add(-1)
			if tok >= e.VocabSize {
				tok = 0
			}
			result[idx] = make([]float32, e.EmbedDim)
			copy(result[idx], e.TokenEmbed[tok])
		}(i, token)
	}
	wg.Wait()

	return result
}

func (e *Embeddings) PositionLookup(positions []int) [][]float32 {
	result := make([][]float32, len(positions))

	var wg sync.WaitGroup
	wg.Add(len(positions))

	for i, pos := range positions {
		go func(idx, p int) {
			defer wg.Add(-1)
			if p >= len(e.PositionEmbed) {
				p = len(e.PositionEmbed) - 1
			}
			result[idx] = make([]float32, e.EmbedDim)
			copy(result[idx], e.PositionEmbed[p])
		}(i, pos)
	}
	wg.Wait()

	return result
}
