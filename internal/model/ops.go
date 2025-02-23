package model

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix multiplication
func matMul(a, b [][]float32) [][]float32 {
	n := len(a)
	m := len(b[0])
	p := len(b)
	result := make([][]float32, n)

	if len(a) == 0 || len(b) == 0 || len(a[0]) != len(b) {
		panic(fmt.Sprintf("matMul dimension mismatch: a(%dx%d), b(%dx%d)", len(a), len(a[0]), len(b), len(b[0])))
	}
	for i := range result {
		result[i] = make([]float32, m)
		for j := range result[i] {
			var sum float32
			for k := 0; k < p; k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

// Vector addition
func addVectors(a, b [][]float32) [][]float32 {
	result := make([][]float32, len(a))
	for i := range a {
		result[i] = make([]float32, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

// Xavier initialization
func xavierInit(shape [][]float32) {
	fanIn := len(shape)
	if fanIn == 0 {
		return
	}

	for i := range shape {
		for j := range shape[i] {
			stddev := float32(math.Sqrt(2.0 / float64(fanIn)))
			shape[i][j] = rand.Float32() * stddev
		}
	}
}
