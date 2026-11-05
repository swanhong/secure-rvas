package workflow

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type prepareGeneSelection struct {
	Mode          string `json:"mode"`
	PerChromosome int    `json:"per_chromosome"`
	Seed          int64  `json:"seed"`
	Path          string `json:"path"`
}

type prepareRequest struct {
	Party            int    `json:"party"`
	NumSites         int    `json:"num_sites"`
	RunDir           string `json:"run_dir"`
	PreparedCacheDir string `json:"prepared_cache_dir"`
	Chromosomes      []int  `json:"chromosomes"`

	Genotype   string `json:"genotype"`
	GenePanel  string `json:"gene_panel"`
	Annotation string `json:"annotation"`
	Phenotype  string `json:"phenotype"`
	Covariate  string `json:"covariate"`
	Ancestry   string `json:"ancestry"`
	Plink2Bin  string `json:"plink2_bin"`

	PhenotypeIDColumn string            `json:"phenotype_id_column"`
	CovariateIDColumn string            `json:"covariate_id_column"`
	CovariateColumn   string            `json:"covariate_column"`
	CovariateColumns  []string          `json:"covariate_columns"`
	IsCovSingleColumn bool              `json:"is_cov_single_column"`
	AncestryIDColumn  string            `json:"ancestry_id_column"`
	AncestryColumn    string            `json:"ancestry_column"`
	PhenotypeColumns  []string          `json:"phenotype_columns"`
	Ancestries        []string          `json:"ancestries"`
	NumCov            int               `json:"num_cov"`
	Mask              map[string]string `json:"mask"`
	MaxMAF            *float64          `json:"max_maf"`

	SamplesPerCohort int64   `json:"samples_per_cohort"`
	SampleSeed       int64   `json:"sample_seed"`
	RoleSeed         int64   `json:"role_seed"`
	SharedRate       float64 `json:"shared_rate"`
	KeepZeroVariants bool    `json:"keep_zero_variants"`

	GeneSelection prepareGeneSelection `json:"gene_selection"`
}

func prepareRequestFromConfig(config *Config) prepareRequest {
	mask := make(map[string]string, len(config.Masks))
	for _, item := range config.Masks {
		column, value, _ := strings.Cut(item, "=")
		mask[strings.TrimSpace(column)] = strings.TrimSpace(value)
	}

	return prepareRequest{
		NumSites:          len(config.Servers) - 1,
		RunDir:            config.RunDir,
		PreparedCacheDir:  config.PreparedCacheDir,
		Chromosomes:       config.Chromosomes,
		Genotype:          config.Genotype,
		GenePanel:         config.GenePanel,
		Annotation:        config.Annotation,
		Phenotype:         config.Phenotype,
		Covariate:         config.Covariate,
		Ancestry:          config.Ancestry,
		Plink2Bin:         config.Plink2Bin,
		PhenotypeIDColumn: config.PhenotypeIDColumn,
		CovariateIDColumn: config.CovariateIDColumn,
		CovariateColumn:   config.CovariateColumn,
		CovariateColumns:  config.CovariateColumns,
		IsCovSingleColumn: config.IsCovSingleColumn,
		AncestryIDColumn:  config.AncestryIDColumn,
		AncestryColumn:    config.AncestryColumn,
		PhenotypeColumns:  config.PhenotypeColumns,
		Ancestries:        config.Ancestries,
		NumCov:            config.NumCov,
		Mask:              mask,
		MaxMAF:            config.MaxMAF,
		SamplesPerCohort:  config.SamplesPerCohort,
		SampleSeed:        config.SampleSeed,
		RoleSeed:          config.RoleSeed,
		SharedRate:        config.SharedRate,
		KeepZeroVariants:  config.KeepZeroVariants,
		GeneSelection: prepareGeneSelection{
			Mode:          config.GeneSelection.Mode,
			PerChromosome: config.GeneSelection.PerChromosome,
			Seed:          config.GeneSelection.Seed,
			Path:          config.GeneSelection.Path,
		},
	}
}

func clearGeneratedOutputs(runDir string) error {
	root, err := filepath.Abs(runDir)
	if err != nil {
		return fmt.Errorf("resolve run directory: %w", err)
	}
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		root = resolved
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("resolve run directory symlinks: %w", err)
	}

	workingDirectory, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("get working directory: %w", err)
	}
	homeDirectory, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("get home directory: %w", err)
	}
	volumeRoot := filepath.VolumeName(root) + string(filepath.Separator)
	if root == volumeRoot || root == workingDirectory || root == homeDirectory {
		return fmt.Errorf("refuse to clear unsafe run directory %s", root)
	}

	for _, name := range []string{
		"selected_genes.tsv",
		"prepared",
		"secure",
		"reference",
		"comparison",
		"metrics",
	} {
		if err := os.RemoveAll(filepath.Join(root, name)); err != nil {
			return fmt.Errorf("remove generated output %s: %w", name, err)
		}
	}
	return nil
}

func Prepare(config *Config) error {
	return prepareSite(config, 0)
}

func prepareSite(config *Config, party int) error {
	request := prepareRequestFromConfig(config)
	request.Party = party
	payload, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("encode preprocessing request: %w", err)
	}

	command := exec.Command(
		"python3",
		"-m",
		"python.preprocessing.prepare",
	)
	command.Stdin = bytes.NewReader(payload)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr

	if err := command.Run(); err != nil {
		return fmt.Errorf("run preprocessing: %w", err)
	}

	return nil
}
