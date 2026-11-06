package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type plamoNode struct {
	next    map[rune]*plamoNode
	id      int
	score   int64
	control bool
}
type plamoTokenizer struct {
	root            plamoNode
	bytes           [256]int
	spaces, repeats int
}

var plamoSpecial = regexp.MustCompile(`<\|plamo:[^|\p{Z}\t\n\r\f\v\x{0085}\x{001C}-\x{001F}]{0,64}\|>`)

func loadPlamo(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	p := &plamoTokenizer{root: plamoNode{id: -1}}
	for i := range p.bytes {
		p.bytes[i] = -1
	}
	t := &Tokenizer{plamo: p, vocab: map[string]int32{}, byteToVal: map[int32]byte{}, special: SpecialTokens{BOS: -1, EOS: -1, Pad: -1, StartOfTurn: -1, EndOfTurn: -1}}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	for scanner.Scan() {
		if len(t.idToPiece) >= maxTokenID {
			return nil, fmt.Errorf("tokenizer(plamo): vocabulary exceeds limit")
		}
		var row []json.RawMessage
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil || len(row) < 3 {
			return nil, fmt.Errorf("tokenizer(plamo): invalid row %d", len(t.idToPiece))
		}
		var piece, kind string
		var score float64
		if json.Unmarshal(row[0], &piece) != nil || json.Unmarshal(row[1], &score) != nil || json.Unmarshal(row[2], &kind) != nil || piece == "" {
			return nil, fmt.Errorf("tokenizer(plamo): invalid vocabulary entry")
		}
		id := len(t.idToPiece)
		t.idToPiece = append(t.idToPiece, piece)
		t.vocab[piece] = int32(id)
		if kind == "BYTE" {
			var value uint8
			if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || !strings.HasSuffix(piece, ">") {
				return nil, fmt.Errorf("tokenizer(plamo): invalid byte token %q", piece)
			}
			if _, err := fmt.Sscanf(piece, "<0x%02X>", &value); err != nil {
				return nil, err
			}
			p.bytes[value] = id
			t.byteToVal[int32(id)] = value
			continue
		}
		node := &p.root
		for _, r := range piece {
			if node.next == nil {
				node.next = map[rune]*plamoNode{}
			}
			if node.next[r] == nil {
				node.next[r] = &plamoNode{id: -1}
			}
			node = node.next[r]
		}
		node.id, node.score, node.control = id, int64(math.RoundToEven(score*1e4)), kind == "CONTROL"
		if node.control {
			t.markAdded(id)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	for _, id := range p.bytes {
		if id < 0 {
			return nil, fmt.Errorf("tokenizer(plamo): incomplete byte fallback vocabulary")
		}
	}
	var cfg struct {
		BOS     string `json:"bos_token"`
		EOS     string `json:"eos_token"`
		Pad     string `json:"pad_token"`
		Spaces  int    `json:"break_around_consecutive_spaces_threshold"`
		Repeats int    `json:"break_around_repeated_chars_threshold"`
	}
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(path), "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	for _, item := range []struct {
		name string
		dst  *int
	}{{cfg.BOS, &t.special.BOS}, {cfg.EOS, &t.special.EOS}, {cfg.Pad, &t.special.Pad}} {
		id, ok := t.vocab[item.name]
		if !ok {
			return nil, fmt.Errorf("tokenizer(plamo): missing special token %q", item.name)
		}
		*item.dst = int(id)
	}
	p.spaces, p.repeats = cfg.Spaces, cfg.Repeats
	if data, err := os.ReadFile(filepath.Join(filepath.Dir(path), "chat_template.jinja")); err == nil {
		t.chatTemplate = string(data)
	}
	return t, nil
}

func (t *Tokenizer) encodePlamo(text string, bos, special bool) []int {
	out := []int{}
	if bos {
		out = append(out, t.special.BOS)
	}
	// Boundary markers are zero-width separators in the reference tokenizer.
	for _, span := range strings.Split(text, "\uee00") {
		start := 0
		for _, loc := range plamoSpecial.FindAllStringIndex(span, -1) {
			out = t.plamoRuns(out, span[start:loc[0]], special)
			out = t.plamoRuns(out, span[loc[0]:loc[1]], special)
			start = loc[1]
		}
		out = t.plamoRuns(out, span[start:], special)
	}
	return out
}

func (t *Tokenizer) plamoRuns(out []int, text string, special bool) []int {
	rs := []rune(text)
	start := 0
	for i := 0; i < len(rs); {
		j := i + 1
		for j < len(rs) && rs[j] == rs[i] {
			j++
		}
		threshold := t.plamo.repeats
		if rs[i] == ' ' && t.plamo.spaces > 0 && (threshold <= 0 || t.plamo.spaces < threshold) {
			threshold = t.plamo.spaces
		}
		if rs[i] != '\n' && threshold > 0 && j-i >= threshold {
			out = t.plamoDP(out, rs[start:i], special)
			out = t.plamoDP(out, rs[i:j], special)
			start = j
		}
		i = j
	}
	return t.plamoDP(out, rs[start:], special)
}

func (t *Tokenizer) plamoDP(out []int, rs []rune, special bool) []int {
	scores := make([]int64, len(rs)+1)
	ids := make([]int, len(rs))
	lengths := make([]int, len(rs))
	for i := len(rs) - 1; i >= 0; i-- {
		scores[i], ids[i], lengths[i] = -(1 << 60), -1, 1
		node := &t.plamo.root
		for j := i; j < len(rs); j++ {
			node = node.next[rs[j]]
			if node == nil {
				break
			}
			if node.id < 0 || node.score <= -20000000 || node.control && !special {
				continue
			}
			candidate := node.score + scores[j+1]
			if candidate >= scores[i] {
				scores[i], ids[i], lengths[i] = candidate, node.id, j-i+1
			}
		}
		if fallback := scores[i+1] - 10000000; fallback > scores[i] {
			scores[i], ids[i], lengths[i] = fallback, -1, 1
		}
	}
	for i := 0; i < len(rs); i += lengths[i] {
		if ids[i] >= 0 {
			out = append(out, ids[i])
		} else {
			for _, b := range []byte(string(rs[i])) {
				out = append(out, t.plamo.bytes[b])
			}
		}
	}
	return out
}

func (t *Tokenizer) decodePlamo(ids []int) (string, error) {
	var out strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.idToPiece) {
			return "", fmt.Errorf("tokenizer(plamo): id %d out of range", id)
		}
		out.Write(t.TokenText(id))
	}
	return out.String(), nil
}
