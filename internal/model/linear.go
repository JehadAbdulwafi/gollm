package model

import (
	"math/rand"
)

type Linear struct {
	InFeatures  int
	OutFeatures int
	Weight      [][]float32
	Bias        []float32
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	l := &Linear{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weight:      make([][]float32, outFeatures),
		Bias:        make([]float32, outFeatures),
	}

	scale := float32(0.02)
	for i := range l.Weight {
		l.Weight[i] = make([]float32, inFeatures)
		for j := range l.Weight[i] {
			l.Weight[i][j] = (rand.Float32()*2 - 1) * scale
		}
	}

	for i := range l.Bias {
		l.Bias[i] = 0
	}

	return l
}

func (l *Linear) Forward(x [][]float32) [][]float32 {
	batchSize := len(x)
	result := make([][]float32, batchSize)

	for b := 0; b < batchSize; b++ {
		result[b] = make([]float32, l.OutFeatures)
		for i := 0; i < l.OutFeatures; i++ {
			sum := l.Bias[i]
			for j := 0; j < l.InFeatures; j++ {
				sum += x[b][j] * l.Weight[i][j]
			}
			result[b][i] = sum
		}
	}

	return result
}
