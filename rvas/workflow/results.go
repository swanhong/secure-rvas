package workflow

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// writeChromosomeResult writes one chromosome's results as soon as it finishes, so a failure later
// in the run (another chromosome, another ancestry, another party) loses nothing already computed.
func writeChromosomeResult(
	runDir string,
	ancestry string,
	result chromosomeResult,
	phenotypeNames []string,
) error {
	secureDir := filepath.Join(runDir, "secure", ancestry)
	if err := os.MkdirAll(secureDir, 0o755); err != nil {
		return err
	}
	return writeSecureResultFile(
		filepath.Join(secureDir, fmt.Sprintf("chr%d.tsv", result.Chromosome)),
		[]chromosomeResult{result},
		phenotypeNames,
	)
}

// writeSecureResults writes the combined file of every chromosome of the ancestry.
func writeSecureResults(
	runDir string,
	ancestry string,
	results []chromosomeResult,
	phenotypeNames []string,
) error {
	secureDir := filepath.Join(runDir, "secure", ancestry)
	if err := os.MkdirAll(secureDir, 0o755); err != nil {
		return err
	}
	return writeSecureResultFile(
		filepath.Join(secureDir, "all_secure_results.tsv"),
		results,
		phenotypeNames,
	)
}

func writeSecureResultFile(
	path string,
	results []chromosomeResult,
	phenotypeNames []string,
) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	writer.Comma = '\t'
	if err := writer.Write([]string{
		"chromosome",
		"gene_index",
		"gene_id",
		"cap",
		"phenotype_index",
		"phenotype_name",
		"secure_burden_p",
		"secure_skat_wh_p",
	}); err != nil {
		file.Close()
		return err
	}

	for _, result := range results {
		for geneIndex, gene := range result.Genes {
			for phenotypeIndex, phenotypeName := range phenotypeNames {
				resultIndex :=
					geneIndex*len(phenotypeNames) +
						phenotypeIndex

				if err := writer.Write([]string{
					strconv.Itoa(result.Chromosome),
					strconv.Itoa(geneIndex),
					gene.GeneID,
					strconv.Itoa(gene.Cap),
					strconv.Itoa(phenotypeIndex),
					phenotypeName,
					strconv.FormatFloat(
						result.BurdenP[resultIndex],
						'g',
						17,
						64,
					),
					strconv.FormatFloat(
						result.SKATWHP[resultIndex],
						'g',
						17,
						64,
					),
				}); err != nil {
					file.Close()
					return err
				}
			}
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}
