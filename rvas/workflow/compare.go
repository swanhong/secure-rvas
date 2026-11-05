package workflow

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/BurntSushi/toml"
)

// compare is a local experiment runner. The protocol itself executes one mode.
func runCompareCommand(args []string) error {
	flags := flag.NewFlagSet("secure-rvas compare", flag.ContinueOnError)
	directory := flags.String("config", "config/1kg-nparty", "source configuration directory")
	output := flags.String("output", "", "new comparison directory (required)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *output == "" || flags.NArg() != 0 {
		return fmt.Errorf("usage: secure-rvas compare --config DIR --output NEW_DIR")
	}
	dirs, err := prepareComparison(*directory, *output)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	for _, dir := range dirs {
		// Each run receives fresh PRG keys; no source key folder is modified.
		if err := GenerateSharedPRGKeys(dir); err != nil {
			return err
		}
		logPath := filepath.Join(filepath.Dir(dir), "run.log")
		log, err := os.Create(logPath)
		if err != nil {
			return err
		}
		fmt.Printf("Running %s; log: %s\n", filepath.Base(filepath.Dir(dir)), logPath)
		command := exec.Command(executable, "run", "--config", dir)
		command.Stdout, command.Stderr = log, log
		err = command.Run()
		log.Close()
		if err != nil {
			return fmt.Errorf("comparison run failed: %w; see %s", err, logPath)
		}
	}
	if err := writeComparisonSummary(dirs, *output); err != nil {
		return err
	}
	fmt.Printf("Comparison complete: %s (dense and hash use the same public caps)\n", *output)
	return nil
}

func prepareComparison(directory, output string) ([]string, error) {
	global, err := loadConfig(directory, globalConfigFilename)
	if err != nil {
		return nil, err
	}
	if err := validateServers(global); err != nil {
		return nil, err
	}
	configs := make([]Config, len(global.Servers))
	for pid := range configs {
		config, err := LoadPartyConfig(directory, pid)
		if err != nil {
			return nil, err
		}
		if err := validatePartyConfig(config, pid); err != nil {
			return nil, err
		}
		configs[pid] = *config
	}
	output, err = filepath.Abs(output)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(output), 0755); err != nil {
		return nil, err
	}
	if err := os.Mkdir(output, 0755); err != nil {
		return nil, err // Never replace an existing comparison or its outputs.
	}
	var dirs []string
	for _, mode := range []string{"dense", "hash"} {
		runDir := filepath.Join(output, mode)
		dir := filepath.Join(runDir, "config")
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}
		for pid, source := range configs {
			config := source
			config.RunDir = runDir
			config.SharedKeysPath = filepath.Join(runDir, "keys", fmt.Sprintf("party%d", pid))
			config.Embedding.Mode = mode
			config.Embedding.CapsDir = "" // Dense computes caps once for this pair.
			if mode == "hash" {
				config.Embedding.CapsDir = filepath.Join(output, "dense", "secure")
			}
			// Local configs contain the resolved source input paths. Changing the
			// output directory must not redirect prepared data to the new run.
			if pid == 0 {
				if err := encodeComparisonConfig(filepath.Join(dir, globalConfigFilename), config); err != nil {
					return nil, err
				}
			}
			if err := encodeComparisonConfig(filepath.Join(dir, fmt.Sprintf("configLocal.Party%d.toml", pid)), config); err != nil {
				return nil, err
			}
		}
		dirs = append(dirs, dir)
	}
	return dirs, nil
}

func encodeComparisonConfig(path string, config Config) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return toml.NewEncoder(f).Encode(config)
}

// Keep party timings separate: these include waits and are not CPU time.
// Keygen spans all M batches; it is not attributed to individual caps.
func writeComparisonSummary(dirs []string, output string) error {
	var header []string
	var rows [][]string
	reference := make(map[string]embeddingPlan)
	for _, dir := range dirs {
		config, err := loadConfig(dir, globalConfigFilename)
		if err != nil {
			return err
		}
		for _, ancestry := range config.Ancestries {
			plans := make(map[string]embeddingPlan)
			for _, chromosome := range config.Chromosomes {
				path := embeddingPlanPath(filepath.Join(config.RunDir, "secure"), ancestry, chromosome)
				plan, err := readEmbeddingPlan(path)
				if err != nil {
					return err
				}
				chr := strconv.Itoa(chromosome)
				key := ancestry + "/" + chr
				if ref, ok := reference[key]; ok && (!slices.Equal(ref.Genes, plan.Genes) || !slices.Equal(ref.Caps, plan.Caps)) {
					return fmt.Errorf("comparison gene/cap mismatch: %s", key)
				}
				reference[key], plans[chr] = plan, plan
			}
			for pid := 0; pid < len(config.Servers); pid++ {
				path := filepath.Join(config.RunDir, "metrics", ancestry, fmt.Sprintf("metrics_party%d.csv", pid))
				f, err := os.Open(path)
				if err != nil {
					return err
				}
				metrics, err := csv.NewReader(f).ReadAll()
				f.Close()
				if err != nil {
					return err
				}
				if len(metrics) == 0 {
					return fmt.Errorf("empty metrics: %s", path)
				}
				header = append([]string{"mode", "M", "rho", "sigma", "H", "genes"}, metrics[0]...)
				for _, metric := range metrics[1:] {
					if !strings.HasPrefix(metric[2], "embedding_") {
						continue
					}
					prefix := []string{config.Embedding.Mode, "", "", "", "", ""}
					genes := 0
					for chr, plan := range plans {
						// Empty chromosome means one M group spanning the ancestry.
						if metric[5] != "" && metric[5] != chr {
							continue
						}
						for _, batch := range plan.Batches {
							p := batch.Setting
							if strings.HasPrefix(metric[2], fmt.Sprintf("embedding_M%d_", p.M)) {
								genes += len(batch.Genes)
								prefix = []string{config.Embedding.Mode, strconv.Itoa(p.M), strconv.Itoa(p.Rho), strconv.Itoa(p.Sigma), strconv.Itoa(p.H), strconv.Itoa(genes)}
							}
						}
					}
					rows = append(rows, append(prefix, metric...))
				}
			}
		}
	}
	return writeCSV(filepath.Join(output, "embedding_comparison.csv"), header, rows)
}
