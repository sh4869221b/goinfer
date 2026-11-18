package main

import "testing"

func TestTranslationSegments(t *testing.T) {
	const text = " a<|plamo:unk|>b "
	const header = "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input"
	const lang = "<|plamo:key|>lang<|plamo:val|>"
	const msg = "<|plamo:msg|>"
	const output = "<|plamo:tag|>output"
	for _, tc := range []struct {
		name, source, target string
		input, tail          string
	}{
		{"easy target", "en", "ja(easy)", header + lang + "en" + msg, output + lang + "ja(easy)" + msg},
		{"easy source", "ja(easy)", "zh-hans", header + lang + "ja(easy)" + msg, output + lang + "zh-hans" + msg},
		{"three-letter language", "jpn", "eng", header + lang + "jpn" + msg, output + lang + "eng" + msg},
		{"unspecified source", "", "ja", header + msg, output + lang + "ja" + msg},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := translationSegments(tc.source, tc.target, text)
			want := []string{tc.input, text, tc.tail}
			if err != nil || len(got) != len(want) {
				t.Fatalf("segments = %+v, %v; want three segments", got, err)
			}
			for i, wantText := range want {
				wantSpecial := i != 1
				if got[i].Text != wantText || got[i].Special != wantSpecial {
					t.Errorf("segment %d = %+v; want %q (special=%v)", i, got[i], wantText, wantSpecial)
				}
			}
		})
	}
	for _, code := range []string{"EN", "en-US", "en-us", "a", "zh-Hans", "zh--hans", "en1", "ja(other)", "ja(easy)<|plamo:tag|>"} {
		if _, err := translationSegments(code, "en", text); err == nil {
			t.Errorf("accepted invalid source language %q", code)
		}
		if _, err := translationSegments("en", code, text); err == nil {
			t.Errorf("accepted invalid target language %q", code)
		}
	}
	if _, err := translationSegments("en", "", text); err == nil {
		t.Error("accepted an empty target language")
	}
}
