package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

func main() {
	inputFile := "data/tiny_shakespeare.txt"
	outputFile := "data/tiny_shakespeare_clean.txt"

	// Read input file
	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Printf("Error opening input file: %v\n", err)
		return
	}
	defer file.Close()

	out, err := os.Create(outputFile)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer out.Close()

	scanner := bufio.NewScanner(file)
	var currentText strings.Builder

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines
		if line == "" {
			continue
		}

		// Skip character names (lines ending with :)
		if strings.HasSuffix(line, ":") {
			continue
		}

		// Skip stage directions (lines in square brackets or parentheses)
		if (strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]")) ||
			(strings.HasPrefix(line, "(") && strings.HasSuffix(line, ")")) {
			continue
		}

		line = cleanText(line)

		if line != "" {
			currentText.WriteString(line + " ")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading file: %v\n", err)
		return
	}

	text := currentText.String()
	text = strings.TrimSpace(text)

	words := strings.Fields(text)
	const wordsPerLine = 20

	for i := 0; i < len(words); i += wordsPerLine {
		end := i + wordsPerLine
		if end > len(words) {
			end = len(words)
		}
		fmt.Fprintf(out, "%s\n", strings.Join(words[i:end], " "))
	}
}

func cleanText(text string) string {
	// Remove multiple spaces
	text = strings.Join(strings.Fields(text), " ")

	// Remove any remaining brackets or parentheses content
	for {
		start := strings.IndexAny(text, "([")
		if start == -1 {
			break
		}
		end := strings.IndexAny(text, ")]")
		if end == -1 {
			break
		}
		text = text[:start] + text[end+1:]
	}

	// Clean punctuation
	text = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) && r != '.' && r != ',' && r != '!' && r != '?' && r != ':' && r != ';' && r != '\'' {
			return -1
		}
		return r
	}, text)

	return strings.TrimSpace(text)
}
