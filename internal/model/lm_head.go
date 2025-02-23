package model

import (
	"math"
	"math/rand"
)

type LMHead struct {
	linear *Linear
}

func NewLMHead(embedDim, vocabSize int) *LMHead {
	return &LMHead{
		linear: NewLinear(embedDim, vocabSize),
	}
}

func (lm *LMHead) Forward(x [][]float32) [][]float32 {
	logits := lm.linear.Forward(x)

	for i := range logits {
		maxLogit := float32(math.Inf(-1))
		for _, v := range logits[i] {
			if v > maxLogit {
				maxLogit = v
			}
		}

		sum := float32(0)
		for j := range logits[i] {
			logits[i][j] = float32(math.Exp(float64(logits[i][j] - maxLogit)))
			sum += logits[i][j]
		}

		for j := range logits[i] {
			logits[i][j] /= sum
		}
	}

	return logits
}

func (lm *LMHead) Sample(probs []float32, temperature float32) int {
	logits := make([]float32, len(probs))
	maxLogit := float32(math.Inf(-1))

	for i, p := range probs {
		if p > 0 {
			logits[i] = float32(math.Log(float64(p))) / temperature
		} else {
			logits[i] = float32(math.Inf(-1))
		}
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	sum := float32(0)
	for i := range logits {
		logits[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		sum += logits[i]
	}

	for i := range logits {
		logits[i] /= sum
	}

	r := rand.Float32()
	cumsum := float32(0)
	for i, p := range logits {
		cumsum += p
		if r < cumsum {
			return i
		}
	}

	maxIdx := 0
	maxProb := float32(-1)
	for i, p := range logits {
		if p > maxProb {
			maxProb = p
			maxIdx = i
		}
	}
	return maxIdx
}
