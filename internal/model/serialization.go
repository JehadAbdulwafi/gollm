package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

// ModelState represents the complete state of a GPT2 model
type ModelState struct {
	Config Config `json:"config"`
	
	// Embeddings
	TokenEmbeddings  [][]float32 `json:"token_embeddings"`
	PositionEmbeddings [][]float32 `json:"position_embeddings"`
	
	// Transformer layers
	Layers []TransformerLayerState `json:"layers"`
	
	// Final normalization
	FinalNormGamma []float32 `json:"final_norm_gamma"`
	FinalNormBeta  []float32 `json:"final_norm_beta"`
	
	// Language model head
	LMHeadWeight [][]float32 `json:"lm_head_weight"`
	LMHeadBias   []float32   `json:"lm_head_bias"`
}

// TransformerLayerState represents the state of a single transformer layer
type TransformerLayerState struct {
	// Self attention
	QKVProjWeight [][]float32 `json:"qkv_proj_weight"`
	QKVProjBias   []float32   `json:"qkv_proj_bias"`
	OutProjWeight [][]float32 `json:"out_proj_weight"`
	OutProjBias   []float32   `json:"out_proj_bias"`
	
	// Layer normalization 1
	Norm1Gamma []float32 `json:"norm1_gamma"`
	Norm1Beta  []float32 `json:"norm1_beta"`
	
	// Feed forward
	FF1Weight [][]float32 `json:"ff1_weight"`
	FF1Bias   []float32   `json:"ff1_bias"`
	FF2Weight [][]float32 `json:"ff2_weight"`
	FF2Bias   []float32   `json:"ff2_bias"`
	
	// Layer normalization 2
	Norm2Gamma []float32 `json:"norm2_gamma"`
	Norm2Beta  []float32 `json:"norm2_beta"`
}

// Save saves model weights to a file
func (g *GPT2) Save(path string) error {
	state := &ModelState{
		Config: g.config,
		TokenEmbeddings: g.embeddings.TokenEmbed,
		PositionEmbeddings: g.embeddings.PositionEmbed,
		Layers: make([]TransformerLayerState, len(g.layers)),
		FinalNormGamma: g.finalNorm.Gamma,
		FinalNormBeta: g.finalNorm.Beta,
		LMHeadWeight: g.lmHead.linear.Weight,
		LMHeadBias: g.lmHead.linear.Bias,
	}

	// Save transformer layer states
	for i, layer := range g.layers {
		state.Layers[i] = TransformerLayerState{
			QKVProjWeight: layer.Attention.QKVProj.Weight,
			QKVProjBias:   layer.Attention.QKVProj.Bias,
			OutProjWeight: layer.Attention.OutProj.Weight,
			OutProjBias:   layer.Attention.OutProj.Bias,
			Norm1Gamma:    layer.Norm1.Gamma,
			Norm1Beta:     layer.Norm1.Beta,
			FF1Weight:     layer.FFN.fc1.Weight,
			FF1Bias:       layer.FFN.fc1.Bias,
			FF2Weight:     layer.FFN.fc2.Weight,
			FF2Bias:       layer.FFN.fc2.Bias,
			Norm2Gamma:    layer.Norm2.Gamma,
			Norm2Beta:     layer.Norm2.Beta,
		}
	}

	// Create file
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer f.Close()

	// Write magic number and version
	if err := binary.Write(f, binary.LittleEndian, uint32(0x476F4C4D)); err != nil { // "GoLM" in hex
		return fmt.Errorf("failed to write magic number: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil { // version 1
		return fmt.Errorf("failed to write version: %v", err)
	}

	// Encode and write model state
	encoder := json.NewEncoder(f)
	if err := encoder.Encode(state); err != nil {
		return fmt.Errorf("failed to encode model state: %v", err)
	}

	return nil
}

// Load loads model weights from a file
func (g *GPT2) Load(path string) error {
	// Open file
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer f.Close()

	// Read and verify magic number
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return fmt.Errorf("failed to read magic number: %v", err)
	}
	if magic != 0x476F4C4D {
		return fmt.Errorf("invalid model file format")
	}

	// Read and verify version
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("failed to read version: %v", err)
	}
	if version != 1 {
		return fmt.Errorf("unsupported model version: %d", version)
	}

	// Decode model state
	var state ModelState
	decoder := json.NewDecoder(f)
	if err := decoder.Decode(&state); err != nil {
		return fmt.Errorf("failed to decode model state: %v", err)
	}

	// Verify config matches
	if state.Config != g.config {
		return fmt.Errorf("model configuration mismatch")
	}

	// Load embeddings
	g.embeddings.TokenEmbed = state.TokenEmbeddings
	g.embeddings.PositionEmbed = state.PositionEmbeddings

	// Load transformer layers
	if len(state.Layers) != len(g.layers) {
		return fmt.Errorf("layer count mismatch")
	}
	for i, layerState := range state.Layers {
		layer := g.layers[i]
		
		// Load attention weights
		layer.Attention.QKVProj.Weight = layerState.QKVProjWeight
		layer.Attention.QKVProj.Bias = layerState.QKVProjBias
		layer.Attention.OutProj.Weight = layerState.OutProjWeight
		layer.Attention.OutProj.Bias = layerState.OutProjBias
		
		// Load layer norms
		layer.Norm1.Gamma = layerState.Norm1Gamma
		layer.Norm1.Beta = layerState.Norm1Beta
		layer.Norm2.Gamma = layerState.Norm2Gamma
		layer.Norm2.Beta = layerState.Norm2Beta
		
		// Load feedforward weights
		layer.FFN.fc1.Weight = layerState.FF1Weight
		layer.FFN.fc1.Bias = layerState.FF1Bias
		layer.FFN.fc2.Weight = layerState.FF2Weight
		layer.FFN.fc2.Bias = layerState.FF2Bias
	}

	// Load final normalization
	g.finalNorm.Gamma = state.FinalNormGamma
	g.finalNorm.Beta = state.FinalNormBeta

	// Load language model head
	g.lmHead.linear.Weight = state.LMHeadWeight
	g.lmHead.linear.Bias = state.LMHeadBias

	return nil
}
