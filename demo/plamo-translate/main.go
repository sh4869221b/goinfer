// Command plamo-translate translates one text using a local PLaMo 3 safetensors checkpoint.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"regexp"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	model := flag.String("model", "", "local PLaMo 3 BF16 safetensors directory")
	source := flag.String("source-lang", "", "source language code (optional, e.g. en)")
	target := flag.String("target-lang", "ja", "target language code (e.g. ja, en)")
	text := flag.String("text", "", "text to translate")
	quant := flag.String("quant", "int4", "runtime weight precision: f32, bf16, int8, int4")
	limit := flag.Int("max-tokens", 256, "maximum number of output tokens")
	flag.Parse()
	if *model == "" || *text == "" || *limit <= 0 {
		return fmt.Errorf("provide -model, nonempty -text and positive -max-tokens")
	}
	language := regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9-]*$`)
	if !language.MatchString(*target) || *source != "" && !language.MatchString(*source) {
		return fmt.Errorf("invalid language code")
	}
	tk, err := tokenizer.Load(*model)
	if err != nil {
		return err
	}
	header := "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input"
	if *source != "" {
		header += "<|plamo:key|>lang<|plamo:val|>" + *source
	}
	header += "<|plamo:msg|>"
	tail := "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + *target + "<|plamo:msg|>"
	prompt, err := tk.EncodeSegments([]tokenizer.Segment{{Text: header, Special: true}, {Text: *text}, {Text: tail, Special: true}}, true)
	if err != nil {
		return err
	}
	m, err := decoder.Load(*model, decoder.Options{Backend: "cpu", Quant: *quant})
	if err != nil {
		return err
	}
	defer m.Close()
	if m.Config().ModelType != "plamo3" {
		return fmt.Errorf("this command requires a PLaMo 3 checkpoint")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	ids, generation := m.Generate(ctx, prompt, *limit, decoder.SamplingParams{})
	stream := tk.NewStream()
	for id := range ids {
		piece, err := stream.Write(id)
		if err != nil {
			cancel()
			for range ids {
			}
			return err
		}
		if _, err := fmt.Print(piece); err != nil {
			cancel()
			for range ids {
			}
			return err
		}
	}
	fmt.Println(stream.Flush())
	return generation.Err()
}
