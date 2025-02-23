package model

import "math"

type FeedForward struct {
	fc1 *Linear
	fc2 *Linear
}

func NewFeedForward(embedDim int) *FeedForward {
	return &FeedForward{
		fc1: NewLinear(embedDim, 4*embedDim),
		fc2: NewLinear(4*embedDim, embedDim),
	}
}

func (ff *FeedForward) Forward(x [][]float32) [][]float32 {
	x = ff.fc1.Forward(x)
	x = gelu(x)
	return ff.fc2.Forward(x)
}

// Gaussian Error Linear Unit approximation
func gelu(x [][]float32) [][]float32 {
	result := make([][]float32, len(x))

	for i := range x {
		result[i] = make([]float32, len(x[i]))
		for j := range x[i] {
			result[i][j] = 0.5 * x[i][j] * (1 + float32(math.Tanh(
				math.Sqrt(2/math.Pi)*(float64(x[i][j])+0.044715*math.Pow(float64(x[i][j]), 3)),
			)))
		}
	}
	return result
}
