package main

import (
	"fmt"
	"os"

	"github.com/hhcho/sfgwas/rvas/workflow"
)

func main() {
	if err := workflow.Main(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "secure-rvas: %v\n", err)
		os.Exit(1)
	}
}
