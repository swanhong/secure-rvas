package workflow

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
	securecrypto "github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/hhcho/sfgwas/rvas/cap"
	"github.com/hhcho/sfgwas/rvas/embed"
)

const (
	globalConfigFilename  = "configGlobal.toml"
	prepareConfigFilename = "configPrepare.toml"
)

type Config struct {
	RunDir           string `toml:"run_dir"`
	PreparedCacheDir string `toml:"prepared_cache_dir"`
	Chromosomes      []int  `toml:"chromosomes"`

	Genotype   string `toml:"genotype"`
	GenePanel  string `toml:"gene_panel"`
	Annotation string `toml:"annotation"`
	Phenotype  string `toml:"phenotype"`
	Covariate  string `toml:"covariate"`
	Ancestry   string `toml:"ancestry"`

	PhenotypeIDColumn string   `toml:"phenotype_id_column"`
	CovariateIDColumn string   `toml:"covariate_id_column"`
	CovariateColumn   string   `toml:"covariate_column"`
	CovariateColumns  []string `toml:"covariate_columns"`
	IsCovSingleColumn bool     `toml:"is_cov_single_column"`
	AncestryIDColumn  string   `toml:"ancestry_id_column"`
	AncestryColumn    string   `toml:"ancestry_column"`
	PhenotypeColumns  []string `toml:"phenotype_columns"`
	Ancestries        []string `toml:"ancestries"`
	NumCov            int      `toml:"num_cov"`
	Masks             []string `toml:"masks"`
	MaxMAF            *float64 `toml:"max_maf"`

	SamplesPerCohort int64   `toml:"samples_per_cohort"`
	SampleSeed       int64   `toml:"sample_seed"`
	RoleSeed         int64   `toml:"role_seed"`
	SharedRate       float64 `toml:"shared_rate"`
	KeepZeroVariants bool    `toml:"keep_zero_variants"`

	CKKS           string          `toml:"ckks"`
	MpcNumThreads  int             `toml:"mpc_num_threads"`
	DataBits       int             `toml:"data_bits"`
	FractionalBits int             `toml:"fractional_bits"`
	Probes         int             `toml:"probes"`
	Seed           int64           `toml:"seed"`
	Embedding      EmbeddingConfig `toml:"embedding"`

	Plink2Bin     string        `toml:"plink2_bin"`
	GeneSelection GeneSelection `toml:"gene_selection"`

	BindingIP string                `toml:"binding_ipaddr"`
	Servers   map[string]mpc.Server `toml:"servers"`

	SharedKeysPath    string `toml:"shared_keys_path"`
	LocalNumThreads   int    `toml:"local_num_threads"`
	GenotypeDirectory string `toml:"genotype_dir"`
	PhenotypeFile     string `toml:"phenotype_file"`
	CovariateFile     string `toml:"covariate_file"`
	GenesFile         string `toml:"genes_file"`
	KeysFile          string `toml:"keys_file"`
}

// Mode selects dense, hash or auto; public caps determine the actual table sizes.
type EmbeddingConfig struct {
	Mode     string           `toml:"mode"`
	DenseMax int              `toml:"dense_max"` // inclusive dense cap limit in auto mode
	Cap      cap.Params       `toml:"cap"`
	Hash     embed.HashPolicy `toml:"hash"`
	CapsDir  string           `toml:"caps_dir"` // optional prior run's secure/ directory
}

func validateServers(config *Config) error {
	if len(config.Servers) < 3 {
		return fmt.Errorf("servers must include P0 and at least two data sites")
	}
	for party := 0; party < len(config.Servers); party++ {
		if _, ok := config.Servers[fmt.Sprintf("party%d", party)]; !ok {
			return fmt.Errorf("servers must be contiguous party0..party%d", len(config.Servers)-1)
		}
	}
	return nil
}

type GeneSelection struct {
	Mode          string `toml:"mode"`
	PerChromosome int    `toml:"per_chromosome"`
	Seed          int64  `toml:"seed"`
	Path          string `toml:"path"`
}

func loadConfig(directory string, filenames ...string) (*Config, error) {
	config := &Config{SharedRate: 0.6, IsCovSingleColumn: true, Embedding: EmbeddingConfig{Mode: "dense", Cap: cap.DefaultParams()}}
	for _, filename := range filenames {
		path := filepath.Join(directory, filename)
		metadata, err := toml.DecodeFile(path, config)
		if err != nil {
			return nil, fmt.Errorf("decode config %q: %w", path, err)
		}
		if metadata.IsDefined("embedding", "tables") {
			return nil, fmt.Errorf("%s: embedding.tables is replaced by embedding.mode and embedding.hash", path)
		}
	}
	for index, ancestry := range config.Ancestries {
		config.Ancestries[index] = strings.ToUpper(strings.TrimSpace(ancestry))
	}
	for _, path := range []*string{
		&config.SharedKeysPath,
		&config.GenotypeDirectory,
		&config.PhenotypeFile,
		&config.CovariateFile,
		&config.GenesFile,
		&config.KeysFile,
	} {
		*path = strings.ReplaceAll(*path, "{run_dir}", config.RunDir)
	}
	return config, nil
}

func LoadPrepareConfig(directory string) (*Config, error) {
	return loadConfig(directory, globalConfigFilename, prepareConfigFilename)
}

func LoadPartyConfig(directory string, partyID int) (*Config, error) {
	return loadConfig(
		directory,
		globalConfigFilename,
		fmt.Sprintf("configLocal.Party%d.toml", partyID),
	)
}

func requireStrings(fields ...string) error {
	for index := 0; index < len(fields); index += 2 {
		if strings.TrimSpace(fields[index+1]) == "" {
			return fmt.Errorf("%s is required", fields[index])
		}
	}
	return nil
}

func validateGlobalConfig(config *Config) error {
	if err := requireStrings("ckks", config.CKKS); err != nil {
		return err
	}
	if len(config.Chromosomes) == 0 || len(config.Ancestries) == 0 || len(config.PhenotypeColumns) == 0 {
		return fmt.Errorf("chromosomes, ancestries, and phenotype_columns are required")
	}
	if config.NumCov < 1 || config.MpcNumThreads < 1 || config.DataBits < 1 ||
		config.FractionalBits < 1 || config.FractionalBits > config.DataBits || config.Probes < 1 {
		return fmt.Errorf("invalid protocol dimensions")
	}
	if !securecrypto.IsPackedSKATParameters(config.CKKS) {
		return fmt.Errorf("unsupported packed SKAT CKKS parameters %q", config.CKKS)
	}
	return nil
}

func validatePrepareConfig(config *Config) error {
	if err := validateGlobalConfig(config); err != nil {
		return err
	}
	if config.SharedRate < 0 || config.SharedRate > 1 {
		return fmt.Errorf("shared_rate must be between 0 and 1")
	}
	if err := validateServers(config); err != nil {
		return err
	}
	if config.IsCovSingleColumn && strings.TrimSpace(config.CovariateColumn) == "" {
		return fmt.Errorf("covariate_column is required")
	}
	if !config.IsCovSingleColumn && len(config.CovariateColumns) != config.NumCov {
		return fmt.Errorf("covariate_columns must contain num_cov columns")
	}
	return requireStrings(
		"run_dir", config.RunDir,
		"genotype", config.Genotype,
		"gene_panel", config.GenePanel,
		"annotation", config.Annotation,
		"phenotype", config.Phenotype,
		"covariate", config.Covariate,
		"ancestry", config.Ancestry,
		"phenotype_id_column", config.PhenotypeIDColumn,
		"covariate_id_column", config.CovariateIDColumn,
		"ancestry_id_column", config.AncestryIDColumn,
		"ancestry_column", config.AncestryColumn,
		"plink2_bin", config.Plink2Bin,
	)
}

func validatePartyConfig(config *Config, partyID int) error {
	if err := validateServers(config); err != nil {
		return err
	}
	if partyID < 0 || partyID >= len(config.Servers) {
		return fmt.Errorf("party ID must be in 0..%d", len(config.Servers)-1)
	}
	if err := validateGlobalConfig(config); err != nil {
		return err
	}
	if err := requireStrings(
		"run_dir", config.RunDir,
		"shared_keys_path", config.SharedKeysPath,
	); err != nil {
		return err
	}
	if _, err := embed.TableSetting(config.Embedding.Mode, 1, 1<<20, config.Embedding.Hash, config.Embedding.DenseMax); err != nil {
		return err
	}
	if config.LocalNumThreads < 1 {
		return fmt.Errorf("local_num_threads must be positive")
	}
	if partyID == auxiliaryPartyID {
		return nil
	}
	return requireStrings(
		"genotype_dir", config.GenotypeDirectory,
		"phenotype_file", config.PhenotypeFile,
		"covariate_file", config.CovariateFile,
		"genes_file", config.GenesFile,
		"keys_file", config.KeysFile,
	)
}

func writeConfigFiles(directory, runDir string, filenames ...string) error {
	destination := filepath.Join(runDir, "config")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		return err
	}
	for _, filename := range filenames {
		contents, err := os.ReadFile(filepath.Join(directory, filename))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(destination, filename), contents, 0o644); err != nil {
			return err
		}
	}
	return nil
}
