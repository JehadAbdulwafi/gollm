package commands

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "gollm",
	Short: "GoLLM - A Go implementation of a Large Language Model",
	Long: `GoLLM is a lightweight implementation of a transformer-based language model in Go.
It supports training, fine-tuning, and text generation.`,
}

// Execute executes the root command
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	// Global flags can be added here
	rootCmd.PersistentFlags().StringP("config", "c", "", "config file (default is ./configs/config.json)")
	rootCmd.AddCommand(generateCmd)
	rootCmd.AddCommand(trainCmd)
	rootCmd.AddCommand(encodeCmd)
}
