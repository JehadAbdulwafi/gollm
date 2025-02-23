package model

type GPT2 struct {
	config     Config
	embeddings *Embeddings
	layers     []*TransformerLayer
	finalNorm  *LayerNorm
	lmHead     *LMHead
}

type Config struct {
	VocabSize   int
	ContextSize int
	EmbedDim    int
	NumHeads    int
	NumLayers   int
}

func NewGPT2(cfg Config) *GPT2 {
	g := &GPT2{
		config:     cfg,
		embeddings: NewEmbeddings(cfg.VocabSize, cfg.EmbedDim, cfg.ContextSize),
		layers:     make([]*TransformerLayer, cfg.NumLayers),
		finalNorm:  NewLayerNorm(cfg.EmbedDim),
		lmHead:     NewLMHead(cfg.EmbedDim, cfg.VocabSize),
	}

	for i := 0; i < cfg.NumLayers; i++ {
		g.layers[i] = NewTransformerLayer(cfg.EmbedDim, cfg.NumHeads)
	}

	return g
}

func (g *GPT2) Forward(input []int) [][]float32 {
	embeddings := g.embeddings.Lookup(input)

	positions := make([]int, len(input))
	for i := range positions {
		positions[i] = i
	}
	posEmbed := g.embeddings.PositionLookup(positions)

	x := make([][]float32, len(embeddings))
	for i := range embeddings {
		x[i] = make([]float32, len(embeddings[i]))
		for j := range embeddings[i] {
			x[i][j] = embeddings[i][j] + posEmbed[i][j]
		}
	}

	for _, layer := range g.layers {
		x = layer.Forward(x)
	}

	x = g.finalNorm.Apply(x)

	return g.lmHead.Forward(x)
}

func (g *GPT2) Loss(input []int, targets []int) float32 {
	logits := g.Forward(input)
	return crossEntropyLoss(logits, targets)
}

func crossEntropyLoss(logits [][]float32, targets []int) float32 {
	return 0.0
}

// Generate generates text given a prompt
func (g *GPT2) Generate(prompt []int, maxTokens int, temperature float32) []int {
	if temperature <= 0 {
		temperature = 0.7
	}

	tokens := make([]int, len(prompt))
	copy(tokens, prompt)

	for len(tokens) < maxTokens {
		context := tokens
		if len(tokens) > g.config.ContextSize {
			context = tokens[len(tokens)-g.config.ContextSize:]
		}

		probs := g.Forward(context)
		nextTokenProbs := probs[len(probs)-1]

		nextToken := g.lmHead.Sample(nextTokenProbs, temperature)

		if nextToken == 0 {
			break
		}

		tokens = append(tokens, nextToken)
	}

	return tokens
}
