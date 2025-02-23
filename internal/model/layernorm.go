package model

import "math"

type LayerNorm struct {
	Gamma []float32
	Beta  []float32
	Eps   float32
}

func NewLayerNorm(dim int) *LayerNorm {
	ln := &LayerNorm{
		Gamma: make([]float32, dim),
		Beta:  make([]float32, dim),
		Eps:   1e-5,
	}

	for i := range ln.Gamma {
		ln.Gamma[i] = 1
	}
	return ln
}

func (ln *LayerNorm) Apply(x [][]float32) [][]float32 {
	output := make([][]float32, len(x))

	for i, vec := range x {
		var mean float32
		for _, v := range vec {
			mean += v
		}
		mean /= float32(len(vec))

		var variance float32
		for _, v := range vec {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float32(len(vec))

		stdDev := float32(math.Sqrt(float64(variance) + float64(ln.Eps)))
		output[i] = make([]float32, len(vec))

		for j, v := range vec {
			normalized := (v - mean) / stdDev
			output[i][j] = normalized*ln.Gamma[j] + ln.Beta[j]
		}
	}
	return output
}
