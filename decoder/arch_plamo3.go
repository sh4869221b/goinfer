package decoder

import (
	"fmt"
	"math"
)

// plamo3Architecture uses folded norm weights and the generic four-norm sandwich.
func plamo3Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if cfg.PlamoLinearType != "" && cfg.PlamoLinearType != "normal" {
		return nil, nil, fmt.Errorf("decoder(plamo3): unsupported linear_type %q; use the original BF16 safetensors", cfg.PlamoLinearType)
	}
	if len(cfg.QuantizationConfig) != 0 && string(cfg.QuantizationConfig) != "null" {
		return nil, nil, fmt.Errorf("decoder(plamo3): pre-quantized safetensors are unsupported")
	}
	if cfg.PlamoWindowSize <= 0 || cfg.SlidingWindowPattern <= 0 {
		return nil, nil, fmt.Errorf("decoder(plamo3): window_size and sliding_window_pattern must be positive")
	}
	full, local, err := parseRopeParameters(cfg.RopeParameters)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder(plamo3): %w", err)
	}
	if full == nil || local == nil || full.base <= 0 || local.base <= 0 {
		return nil, nil, fmt.Errorf("decoder(plamo3): full_attention and sliding_attention rope_parameters are required")
	}
	if len(cfg.LayerTypes) != 0 && len(cfg.LayerTypes) != cfg.NumLayers {
		return nil, nil, fmt.Errorf("decoder(plamo3): layer_types length must equal num_hidden_layers")
	}
	for _, kind := range cfg.LayerTypes {
		if kind != "full_attention" && kind != "sliding_attention" {
			return nil, nil, fmt.Errorf("decoder(plamo3): unsupported layer type %q", kind)
		}
	}
	hd := cfg.HeadDim
	if hd <= 0 || hd%2 != 0 {
		return nil, nil, fmt.Errorf("decoder(plamo3): head_dim must be positive and even")
	}
	if cfg.IntermediateDim <= 0 {
		return nil, nil, fmt.Errorf("decoder(plamo3): intermediate_size must be positive")
	}
	scale := 1.0
	if cfg.PlamoScaleEmbedding {
		scale = math.Sqrt(float64(cfg.HiddenDim))
	}
	return &Architecture{
		Name: "plamo3", HiddenDim: cfg.HiddenDim, NumLayers: cfg.NumLayers,
		NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: hd,
		IntermediateDim: cfg.IntermediateDim, VocabSize: cfg.VocabSize,
		Norm: NormRMS, NormEps: cfg.RMSNormEps, NormPlacement: NormSandwich4,
		Act: ActSiLU, QKNorm: true, AttnScale: math.Pow(float64(hd), -0.5),
		SlidingWindow: cfg.PlamoWindowSize, layerIsGlobal: cfg.IsGlobalLayer,
		RoPEGlobalBase: full.base, RoPELocalBase: local.base,
		ropeScaling: full.scaling, ropeScalingLocal: local.scaling,
		EmbedScale: scale, TiedLMHead: cfg.TieWordEmbeddings,
	}, &plamo3TensorSchema, nil
}

var plamo3TensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}
