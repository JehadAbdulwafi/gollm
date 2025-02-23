package model

type TransformerLayer struct {
	Attention *MultiHeadAttention
	FFN       *FeedForward
	Norm1     *LayerNorm
	Norm2     *LayerNorm
}

func NewTransformerLayer(embedDim, numHeads int) *TransformerLayer {
	return &TransformerLayer{
		Attention: NewMultiHeadAttention(embedDim, numHeads),
		FFN:       NewFeedForward(embedDim),
		Norm1:     NewLayerNorm(embedDim),
		Norm2:     NewLayerNorm(embedDim),
	}
}

func (l *TransformerLayer) Forward(x [][]float32) [][]float32 {
	// Self-attention with residual connection
	attnOut := l.Attention.Forward(x)
	residual := addVectors(x, attnOut)
	norm1Out := l.Norm1.Apply(residual)

	// Feed-forward with residual connection
	ffnOut := l.FFN.Forward(norm1Out)
	residual = addVectors(norm1Out, ffnOut)
	return l.Norm2.Apply(residual)
}

