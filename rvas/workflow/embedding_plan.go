package workflow

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/hhcho/sfgwas/rvas/embed"
)

type embeddingPlan struct {
	Genes          []string
	Caps           []int
	Batches        []embed.EmbeddingBatch
	GaloisCount    int
	GaloisKeyBytes int
	KeyScope       string // "ancestry": the same shared key context, not an additional allocation per chromosome
}

func embeddingPlanPath(directory, ancestry string, chromosome int) string {
	return filepath.Join(directory, ancestry, fmt.Sprintf("embedding_chr%d.json", chromosome))
}

func readEmbeddingPlan(path string) (embeddingPlan, error) {
	var plan embeddingPlan
	data, err := os.ReadFile(path)
	if err != nil {
		return plan, err
	}
	if err := json.Unmarshal(data, &plan); err != nil {
		return plan, fmt.Errorf("%s: %w", path, err)
	}
	return plan, nil
}

func loadEmbeddingCaps(directory, ancestry string, partyID int, input *PartyInput) error {
	if directory == "" {
		return nil
	}
	for i := range input.Chromosomes {
		chromosome := &input.Chromosomes[i]
		path := embeddingPlanPath(directory, ancestry, chromosome.Chromosome)
		plan, err := readEmbeddingPlan(path)
		if err != nil {
			return err
		}
		if plan.Caps == nil || len(plan.Caps) != len(plan.Genes) {
			return fmt.Errorf("%s: missing caps or inconsistent gene count", path)
		}
		if partyID != 0 {
			genes := make([]string, len(chromosome.Genes))
			for j, gene := range chromosome.Genes {
				genes[j] = gene.GeneID
			}
			if !slices.Equal(genes, plan.Genes) {
				return fmt.Errorf("%s: gene order differs from prepared input", path)
			}
		}
		chromosome.PublicCaps = plan.Caps
	}
	return nil
}

func writeEmbeddingPlans(runDir, ancestry string, input *PartyInput, results []embed.PipelineResult) error {
	for i, chromosome := range input.Chromosomes {
		result := results[i]
		plan := embeddingPlan{
			Genes: make([]string, len(chromosome.Genes)), Caps: result.Caps,
			Batches: result.Batches, GaloisCount: len(result.Galois), GaloisKeyBytes: result.GaloisKeyBytes,
			KeyScope: "ancestry",
		}
		for g, gene := range chromosome.Genes {
			plan.Genes[g] = gene.GeneID
		}
		data, err := json.MarshalIndent(plan, "", "  ")
		if err != nil {
			return err
		}
		path := embeddingPlanPath(filepath.Join(runDir, "secure"), ancestry, chromosome.Chromosome)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}
		if err := os.WriteFile(path, append(data, '\n'), 0644); err != nil {
			return err
		}
	}
	return nil
}
