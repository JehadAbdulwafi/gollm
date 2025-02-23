package tokenizer

import (
	"strings"
	"unicode"
)

const (
	defaultPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

type Preprocessor struct {
	punctuation map[rune]bool
	lowercase   bool
}

func NewPreprocessor(lowercase bool) *Preprocessor {
	p := &Preprocessor{
		lowercase:   lowercase,
		punctuation: make(map[rune]bool),
	}

	for _, r := range defaultPunctuation {
		p.punctuation[r] = true
	}

	return p
}

func (p *Preprocessor) Clean(text string) string {
	if p.lowercase {
		text = strings.ToLower(text)
	}

	return strings.Map(func(r rune) rune {
		if p.punctuation[r] || unicode.IsSpace(r) {
			return ' '
		}
		return r
	}, text)
}

func (p *Preprocessor) NormalizeWhitespace(text string) string {
	return strings.Join(strings.Fields(text), " ")
}