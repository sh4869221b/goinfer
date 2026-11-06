package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// foldPlamoNorm makes the norm's offset part of its weight, including on the GPU paths.
func foldPlamoNorm(st *embed.SafetensorsFile, name string, size int, offset float32) ([]float32, error) {
	data, err := st.TensorF32(name, size)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = v + offset
	}
	return out, nil
}

func buildPlamo3Weights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, sink *giwWriter, id string) (*Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	// q is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for every layer projection below. Embed/LMHead stay on plain
	// quantizeWM: they are read via .Row() on the host on every backend.
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return quantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, "model.embed_tokens.weight", vocab, hidden, quant.embeddingWith(embedInt4), needCanonical, sink); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = foldPlamoNorm(st, "model.norm.weight", hidden, 1); err != nil {
		return nil, err
	}
	if !arch.TiedLMHead {
		if w.LMHead, err = loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.embeddingWith(embedInt4), needCanonical, sink); err != nil {
			return nil, err
		}
	}

	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("model.layers.layers.%d.", i)
		var e error
		for _, norm := range []struct {
			dst    *[]float32
			name   string
			size   int
			offset float32
		}{
			{&l.PreAttnNorm, "pre_mixer_norm.weight", hidden, 1},
			{&l.PostAttnNorm, "post_mixer_norm.weight", hidden, 0.2},
			{&l.PreMLPNorm, "pre_mlp_norm.weight", hidden, 1},
			{&l.PostMLPNorm, "post_mlp_norm.weight", hidden, float32(1 / math.Pow(5, 1.5))},
			{&l.QNorm, "mixer.q_norm.weight", hd, 1},
			{&l.KNorm, "mixer.k_norm.weight", hd, 1},
		} {
			if *norm.dst, e = foldPlamoNorm(st, p+norm.name, norm.size, norm.offset); e != nil {
				return e
			}
		}
		// Fused qkv_proj [qDim+2*kvDim, hidden] → Q ‖ K ‖ V by output rows.
		qkv, qerr := st.TensorF32(p+"mixer.qkv_proj.weight", qDim+2*kvDim, hidden)
		if qerr != nil {
			return qerr
		}
		l.QProj = q(linalg.WrapF32(qkv[0:qDim*hidden], qDim, hidden))
		l.KProj = q(linalg.WrapF32(qkv[qDim*hidden:(qDim+kvDim)*hidden], kvDim, hidden))
		l.VProj = q(linalg.WrapF32(qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden], kvDim, hidden))
		if l.OProj, e = loadMat(st, p+"mixer.o_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.OProj = q(l.OProj)
		// Fused gate_up_proj [2*inter, hidden] → gate ‖ up (SwiGLU: down(silu(gate)·up)).
		gu, gerr := st.TensorF32(p+"mlp.gate_up_proj.weight", 2*inter, hidden)
		if gerr != nil {
			return gerr
		}
		l.GateProj = q(linalg.WrapF32(gu[0:inter*hidden], inter, hidden))
		l.UpProj = q(linalg.WrapF32(gu[inter*hidden:2*inter*hidden], inter, hidden))
		if l.DownProj, e = loadMat(st, p+"mlp.down_proj.weight", hidden, inter); e != nil {
			return e
		}
		l.DownProj = q(l.DownProj)
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id)
}
