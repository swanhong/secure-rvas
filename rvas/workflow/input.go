package workflow

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/hhcho/sfgwas/rvas/protocol"
	"gonum.org/v1/gonum/mat"
)

const (
	auxiliaryPartyID = iota
	cohortAPartyID
	cohortBPartyID
)

type ChromosomeInput struct {
	PublicCaps        []int // optional replay of previously revealed caps
	Chromosome        int
	GenotypeDirectory string
	Genes             []protocol.Gene
	Keys              [][]string // per gene, the site's variant keys in block column order
	CryptoParams      protocol.CryptoParams
}

type PartyInput struct {
	X           *mat.Dense
	Y           *mat.Dense
	SampleCount int
	Chromosomes []ChromosomeInput
}

func readTextMatrix(path string, columns int) (*mat.Dense, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	fields := strings.Fields(string(data))
	values := make([]float64, len(fields))
	for index, field := range fields {
		values[index], err = strconv.ParseFloat(field, 64)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}
	if columns < 1 {
		return nil, fmt.Errorf("%s: column count must be positive, got %d", path, columns)
	}
	if len(values) == 0 || len(values)%columns != 0 {
		return nil, fmt.Errorf(
			"%s: %d values do not fill rows of %d columns (prepared with a different column count, or truncated)",
			path, len(values), columns,
		)
	}
	return mat.NewDense(len(values)/columns, columns, values), nil
}

func readRows(
	covariatePath, phenotypePath string,
	covariateCount, phenotypeCount int,
) (*mat.Dense, *mat.Dense, error) {
	covariates, err := readTextMatrix(covariatePath, covariateCount)
	if err != nil {
		return nil, nil, err
	}
	y, err := readTextMatrix(phenotypePath, phenotypeCount)
	if err != nil {
		return nil, nil, err
	}

	rows, _ := covariates.Dims()
	if yRows, _ := y.Dims(); yRows != rows {
		return nil, nil, fmt.Errorf("covariate and phenotype sample counts differ: %d vs %d", rows, yRows)
	}
	x := mat.NewDense(rows, covariateCount+1, nil)
	for row := 0; row < rows; row++ {
		x.Set(row, 0, 1)
		for col := 0; col < covariateCount; col++ {
			x.Set(row, col+1, covariates.At(row, col))
		}
	}
	return x, y, nil
}

// readGenes reads the gene order and the site's variant keys: line i of the keys file holds gene
// i's keys separated by tabs, in block column order (an empty line for a gene without variants).
// A key repeated within a gene is an error, since the matching assumes distinct keys. The gene
// caps are set later, by the alignment.
func readGenes(genePath, keysPath string) ([]protocol.Gene, [][]string, error) {
	geneData, err := os.ReadFile(genePath)
	if err != nil {
		return nil, nil, err
	}
	keyData, err := os.ReadFile(keysPath)
	if err != nil {
		return nil, nil, err
	}

	geneIDs := strings.Fields(string(geneData))
	var lines []string
	if len(keyData) > 0 {
		lines = strings.Split(strings.TrimSuffix(string(keyData), "\n"), "\n")
	}
	if len(geneIDs) != len(lines) {
		return nil, nil, fmt.Errorf("%d genes but %d key lines", len(geneIDs), len(lines))
	}
	genes := make([]protocol.Gene, len(geneIDs))
	keys := make([][]string, len(geneIDs))
	for index, geneID := range geneIDs {
		if lines[index] != "" {
			keys[index] = strings.Split(lines[index], "\t")
		}
		seen := make(map[string]bool, len(keys[index]))
		for _, key := range keys[index] {
			if seen[key] {
				return nil, nil, fmt.Errorf("%s: gene %s lists a key twice", keysPath, geneID)
			}
			seen[key] = true
		}
		genes[index] = protocol.Gene{GeneID: geneID}
	}
	return genes, keys, nil
}

func decodeInt8(data []byte) []float64 {
	values := make([]float64, len(data))
	for index, value := range data {
		values[index] = float64(int8(value))
	}
	return values
}

func readInt8Matrix(
	path string,
	rows, columns int,
) (*mat.Dense, error) {
	if columns == 0 {
		return nil, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return mat.NewDense(
		rows,
		columns,
		decodeInt8(data),
	), nil
}

// readGeneBatch reads each gene's block (the site's own variants, len(keys) columns) and lays it out in
// the public width like the padded rows of the matching step: the first columns up to the width, then
// zero columns.
func readGeneBatch(
	genotypeDirectory string,
	genes []protocol.Gene,
	keys [][]string,
	batch protocol.GeneBatch,
	rows int,
) (genotypes []*mat.Dense, err error) {
	if genotypeDirectory == "" {
		return nil, nil
	}

	genotypes = make([]*mat.Dense, len(genes))
	for _, geneIndex := range batch.GeneIndices {
		local, err := readInt8Matrix(
			filepath.Join(genotypeDirectory, fmt.Sprintf("block.%d.bin", geneIndex)),
			rows,
			len(keys[geneIndex]),
		)
		if err != nil {
			return nil, err
		}
		genotypes[geneIndex] = mat.NewDense(rows, genes[geneIndex].Cap, nil)
		if local != nil {
			genotypes[geneIndex].Copy(local) // copies min(local, public) columns
		}
	}
	return genotypes, nil
}

// readDosageSums returns the site's ALT allele count of each padded row, in its own row order, gene
// after gene in cap-wide blocks; rows past the site's own keys are zero.
func readDosageSums(
	directory string,
	genes []protocol.Gene,
	keys [][]string,
	rows int,
	observe func(string) func(),
) (*mat.Dense, error) {
	total := 0
	for _, gene := range genes {
		total += gene.Cap
	}
	if total == 0 {
		return nil, nil
	}

	dosageSums := make([]float64, total)
	offset := 0
	for geneIndex, gene := range genes {
		width := len(keys[geneIndex]) // the block's own columns
		if width > 0 {
			doneRead := observe("dosage_loading")
			data, err := os.ReadFile(filepath.Join(
				directory,
				fmt.Sprintf("block.%d.bin", geneIndex),
			))
			doneRead()
			if err != nil {
				return nil, err
			}

			doneLocal := observe("local_dosage_sums")
			for row := 0; row < rows; row++ {
				rowOffset := row * width
				for variant := 0; variant < min(width, gene.Cap); variant++ {
					dosageSums[offset+variant] +=
						float64(int8(data[rowOffset+variant]))
				}
			}
			doneLocal()
		}
		offset += gene.Cap
	}

	return mat.NewDense(1, total, dosageSums), nil
}

func loadPartyInput(
	config *Config,
	partyID int,
	ancestry string,
) (*PartyInput, error) {
	if partyID < auxiliaryPartyID || partyID >= len(config.Servers) {
		return nil, fmt.Errorf("unknown party %d", partyID)
	}

	input := &PartyInput{
		Chromosomes: make(
			[]ChromosomeInput,
			len(config.Chromosomes),
		),
	}
	for index, chromosome := range config.Chromosomes {
		input.Chromosomes[index].Chromosome = chromosome
	}

	if partyID == auxiliaryPartyID {
		return input, nil
	}

	firstChromosome := config.Chromosomes[0]
	var err error
	input.X, input.Y, err = readRows(
		expandInputPath(config.CovariateFile, ancestry, firstChromosome),
		expandInputPath(config.PhenotypeFile, ancestry, firstChromosome),
		config.NumCov,
		len(config.PhenotypeColumns),
	)
	if err != nil {
		return nil, err
	}
	input.SampleCount, _ = input.X.Dims()

	for index, chromosome := range config.Chromosomes {
		genes, keys, err := readGenes(
			expandInputPath(config.GenesFile, ancestry, chromosome),
			expandInputPath(config.KeysFile, ancestry, chromosome),
		)
		if err != nil {
			return nil, err
		}

		directory := expandInputPath(config.GenotypeDirectory, ancestry, chromosome)
		// Check dimensions before opening the network, including zero-column blocks.
		for g := range genes {
			path := filepath.Join(directory, fmt.Sprintf("block.%d.bin", g))
			info, err := os.Stat(path)
			if err != nil {
				return nil, err
			}
			expected := int64(input.SampleCount) * int64(len(keys[g]))
			if info.Size() != expected {
				return nil, fmt.Errorf("%s: %d bytes; expected %d", path, info.Size(), expected)
			}
		}
		input.Chromosomes[index] = ChromosomeInput{
			Chromosome:        chromosome,
			GenotypeDirectory: directory,
			Genes:             genes,
			Keys:              keys,
		}
	}

	return input, nil
}

func expandInputPath(path, ancestry string, chromosome int) string {
	path = strings.ReplaceAll(path, "{ancestry}", ancestry)
	return strings.ReplaceAll(path, "{chromosome}", strconv.Itoa(chromosome))
}
