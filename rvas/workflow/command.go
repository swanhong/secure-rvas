package workflow

import (
	"flag"
	"fmt"
)

func Main(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf(
			"usage: secure-rvas <prepare|keygen|run|compare|party|capstats> [--config directory]",
		)
	}

	switch args[0] {
	case "prepare":
		return runPrepareCommand(args[1:])
	case "capstats":
		return runCapstatsCommand(args[1:])
	case "keygen":
		return runKeygenCommand(args[1:])
	case "compare":
		return runCompareCommand(args[1:])
	case "run":
		return runRunCommand(args[1:])
	case "party":
		return runPartyCommand(args[1:])
	default:
		return fmt.Errorf(
			"unknown command %q; expected <prepare|keygen|run|compare|party|capstats>",
			args[0],
		)
	}
}

func runPrepareCommand(args []string) error {
	flags := flag.NewFlagSet(
		"secure-rvas prepare",
		flag.ContinueOnError,
	)
	configDirectory := flags.String(
		"config",
		"config/1kg-nparty",
		"configuration directory",
	)
	clearOutputs := flags.Bool(
		"clear",
		false,
		"remove generated outputs before preprocessing",
	)

	party := flags.Int("party", 0, "0=split common input among all configured data sites; 1..N=prepare one site's own input")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse prepare arguments: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf(
			"unexpected prepare arguments: %v",
			flags.Args(),
		)
	}

	config, err := LoadPrepareConfig(*configDirectory)
	if err != nil {
		return err
	}

	if err := validatePrepareConfig(config); err != nil {
		return fmt.Errorf("validate config: %w", err)
	}
	if *party < 0 || *party >= len(config.Servers) {
		return fmt.Errorf("prepare: --party must be in 0..%d", len(config.Servers)-1)
	}
	if *clearOutputs {
		if err := clearGeneratedOutputs(config.RunDir); err != nil {
			return err
		}
		fmt.Printf("Cleared generated outputs from %s\n", config.RunDir)
	}

	if err := writeConfigFiles(
		*configDirectory,
		config.RunDir,
		globalConfigFilename,
		prepareConfigFilename,
	); err != nil {
		return fmt.Errorf("write config: %w", err)
	}

	fmt.Println("Running secure-rvas::prepare")
	fmt.Printf("Read configuration from %s\n", *configDirectory)
	fmt.Printf("Saved configuration to %s/config\n", config.RunDir)
	if *party == 0 {
		fmt.Printf("Data sites: %d (excluding P0)\n", len(config.Servers)-1)
		fmt.Printf("Shared variant rate: %g\n", config.SharedRate)
	} else {
		fmt.Printf("Preparing Party %d only\n", *party)
	}

	return prepareSite(config, *party)
}

func runRunCommand(args []string) error {
	flags := flag.NewFlagSet(
		"secure-rvas run",
		flag.ContinueOnError,
	)
	configDirectory := flags.String(
		"config",
		"config/1kg-nparty",
		"configuration directory",
	)

	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse run arguments: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf(
			"unexpected run arguments: %v",
			flags.Args(),
		)
	}

	fmt.Println("Running secure-rvas::run")
	fmt.Printf("Read configuration from %s\n", *configDirectory)

	return Run(*configDirectory)
}
