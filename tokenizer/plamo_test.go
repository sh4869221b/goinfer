package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestPlamoTokenizer(t *testing.T) {
	dir := t.TempDir()
	var rows strings.Builder
	vocab := [][]any{{"<|plamo:bos|>", 0, "CONTROL"}, {"<|plamo:pad|>", 0, "CONTROL"}, {"a", -1, "NORMAL"}, {"aa", -2, "NORMAL"}, {"▁", -1, "NORMAL"}, {" ", -1, "NORMAL"}, {"a  a", 0, "NORMAL"}}
	for i := range 256 {
		vocab = append(vocab, []any{fmt.Sprintf("<0x%02X>", i), -100, "BYTE"})
	}
	for _, row := range vocab {
		raw, _ := json.Marshal(row)
		rows.Write(raw)
		rows.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.jsonl"), []byte(rows.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"bos_token":"<|plamo:bos|>","eos_token":"<|plamo:bos|>","pad_token":"<|plamo:pad|>","break_around_consecutive_spaces_threshold":2,"break_around_repeated_chars_threshold":4}`), 0600); err != nil {
		t.Fatal(err)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		text string
		want []int
	}{{"aa", []int{3}}, {"a  a", []int{2, 5, 5, 2}}, {"▁", []int{4}}, {"a\uee00a", []int{2, 2}}} {
		ids, err := tk.Encode(test.text, false)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, test.want) {
			t.Errorf("%q: %v, want %v", test.text, ids, test.want)
		}
	}
	for _, text := range []string{"🙂𠮷", " aa ▁ ", "<|plamo:bos|>a"} {
		ids, err := tk.Encode(text, false)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := tk.Decode(ids)
		if err != nil || decoded != text {
			t.Fatalf("roundtrip %q => %q (%v)", text, decoded, err)
		}
		stream := tk.NewStream()
		var got strings.Builder
		for _, id := range ids {
			s, err := stream.Write(id)
			if err != nil {
				t.Fatal(err)
			}
			got.WriteString(s)
		}
		got.WriteString(stream.Flush())
		if got.String() != text {
			t.Fatalf("stream: %q != %q", got.String(), text)
		}
	}
	ids, err := tk.EncodeSegments([]Segment{{Text: "<|plamo:bos|>", Special: false}}, false)
	if err != nil || slices.Contains(ids, 0) {
		t.Fatalf("content forged control token: %v (%v)", ids, err)
	}
}

func TestPlamoOfficialTokenizerParity(t *testing.T) {
	dir := os.Getenv("PLAMO3_TOKENIZER_DIR")
	if dir == "" {
		t.Skip("set PLAMO3_TOKENIZER_DIR to the pinned official tokenizer directory")
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../testdata/plamo3_tokenizer_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text   string
		IDs    []int
		BOSIDs []int `json:"bos_ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		for _, bos := range []bool{false, true} {
			got, err := tk.Encode(c.Text, bos)
			if err != nil {
				t.Fatal(err)
			}
			want := c.IDs
			if bos {
				want = c.BOSIDs
			}
			if !slices.Equal(got, want) {
				t.Fatalf("bos=%v %q\ngot %v\nwant %v", bos, c.Text, got, want)
			}
		}
	}
	t.Logf("exact PFN token ID parity on %d cases with and without BOS", len(cases))
}
