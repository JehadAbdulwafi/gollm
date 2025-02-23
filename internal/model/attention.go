package model

import (
	"fmt"
	"math"
)

type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	QKVProj  *Linear
	OutProj  *Linear
}

func NewMultiHeadAttention(embedDim, numHeads int) *MultiHeadAttention {
	headDim := embedDim / numHeads
	if embedDim%numHeads != 0 {
		panic(fmt.Sprintf("embedDim (%d) must be divisible by numHeads (%d)", embedDim, numHeads))
	}
	return &MultiHeadAttention{
		NumHeads: numHeads,
		HeadDim:  headDim,
		QKVProj:  NewLinear(embedDim, 3*embedDim),
		OutProj:  NewLinear(embedDim, embedDim),
	}
}

func (mha *MultiHeadAttention) Forward(x [][]float32) [][]float32 {
	batchSize := len(x)
	embedDim := len(x[0])

	qkv := mha.QKVProj.Forward(x)

	q := make([][]float32, batchSize)
	k := make([][]float32, batchSize)
	v := make([][]float32, batchSize)

	for i := range qkv {
		q[i] = make([]float32, embedDim)
		k[i] = make([]float32, embedDim)
		v[i] = make([]float32, embedDim)
		copy(q[i], qkv[i][:embedDim])
		copy(k[i], qkv[i][embedDim:2*embedDim])
		copy(v[i], qkv[i][2*embedDim:])
	}

	output := make([][]float32, batchSize)
	for i := range output {
		output[i] = make([]float32, embedDim)
	}

	for h := 0; h < mha.NumHeads; h++ {
		start := h * mha.HeadDim
		end := (h + 1) * mha.HeadDim
		for b := range output {
			qh := q[b][start:end]
			scores := make([]float32, batchSize)
			scale := 1.0 / float32(math.Sqrt(float64(mha.HeadDim)))

			for i := 0; i < batchSize; i++ {
				kh := k[i][start:end]
				sum := float32(0)
				for j := 0; j < mha.HeadDim; j++ {
					sum += qh[j] * kh[j]
				}
				scores[i] = sum * scale
			}

			maxScore := float32(math.Inf(-1))
			for _, score := range scores {
				if score > maxScore {
					maxScore = score
				}
			}

			expSum := float32(0)
			expScores := make([]float32, batchSize)
			for i, score := range scores {
				expScores[i] = float32(math.Exp(float64(score - maxScore)))
				expSum += expScores[i]
			}

			for j := 0; j < mha.HeadDim; j++ {
				sum := float32(0)
				for i := 0; i < batchSize; i++ {
					vh := v[i][start:end]
					sum += expScores[i] * vh[j] / expSum
				}
				output[b][start+j] = sum
			}
		}
	}

	return mha.OutProj.Forward(output)
}
