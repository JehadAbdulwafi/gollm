package model

import (
	"math"
)

// CrossEntropyLoss calculates the cross entropy loss between logits and target indices
func CrossEntropyLoss(logits [][]float32, targets []int) float32 {
	var loss float32
	batchSize := len(logits)
	vocabSize := len(logits[0])

	for i := 0; i < batchSize; i++ {
		currentLogits := logits[i]
		target := targets[i]

		maxLogit := float32(math.Inf(-1))
		for _, l := range currentLogits {
			if l > maxLogit {
				maxLogit = l
			}
		}

		sumExp := float32(0)
		for _, l := range currentLogits {
			sumExp += float32(math.Exp(float64(l - maxLogit)))
		}

		logSumExp := float32(math.Log(float64(sumExp))) + maxLogit

		if target >= 0 && target < vocabSize {
			loss += logSumExp - currentLogits[target]
		}
	}

	return loss / float32(batchSize)
}
