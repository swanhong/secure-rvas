package workflow

import (
	"flag"
	"fmt"
	"io"
	mathrand "math/rand"
	"os"
	"path/filepath"
	"slices"
	"text/tabwriter"

	"github.com/hhcho/sfgwas/rvas/cap"
)

// capStat is one gene's variant counts and cap. aOnly, bOnly and both partition the union.
type capStat struct {
	chromosome int
	geneID     string
	aOnly      int
	bOnly      int
	both       int
	union      int
	capM       int
}

// runCapstatsCommand prints, per chromosome, how the real union of the two sites' variant keys
// compares with the cap, and how many genes fall into each cap. It reads the output of a split
// `prepare` and does no matching, embedding or MPC. Aggregates go to stdout; per-gene counts go to
// <run_dir>/capstats.tsv.
func runCapstatsCommand(args []string) error {
	flags := flag.NewFlagSet("secure-rvas capstats", flag.ContinueOnError)
	configDirectory := flags.String("config", "config/1kg-nparty", "configuration directory")
	seed := flags.Int64("seed", 1, "seed of the cap hash key (the real run draws a fresh secret key)")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse capstats arguments: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected capstats arguments: %v", flags.Args())
	}
	config, err := LoadPrepareConfig(*configDirectory)
	if err != nil {
		return err
	}
	if err := validatePrepareConfig(config); err != nil {
		return fmt.Errorf("validate config: %w", err)
	}

	capKey := make([]byte, 32)
	mathrand.New(mathrand.NewSource(*seed)).Read(capKey)

	outPath := filepath.Join(config.RunDir, "capstats.tsv")
	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()
	fmt.Fprintln(out, "ancestry\tchromosome\tgene_id\ta_only\tb_only\tintersection\tunion\tcap")

	for _, ancestry := range config.Ancestries {
		directory := filepath.Join(config.RunDir, "prepared", ancestry)
		stats, err := collectCapStats(directory, config.Chromosomes, capKey)
		if err != nil {
			return err
		}
		for _, s := range stats {
			fmt.Fprintf(out, "%s\tchr%d\t%s\t%d\t%d\t%d\t%d\t%d\n",
				ancestry, s.chromosome, s.geneID, s.aOnly, s.bOnly, s.both, s.union, s.capM)
		}
		fmt.Printf("\n######## ancestry %s (cap hash key seed %d) ########\n", ancestry, *seed)
		printCapStats(os.Stdout, stats, config.Chromosomes)
	}
	fmt.Printf("\nPer-gene counts: %s\n", outPath)
	return nil
}

// collectCapStats reads the prepared keys of every chromosome and computes each gene's cap from the
// union sketch of both sites, as the secure caps step would.
func collectCapStats(directory string, chromosomes []int, capKey []byte) ([]capStat, error) {
	hp := cap.DefaultParams()
	th := cap.BuildThresholds(hp)
	var stats []capStat
	for _, chromosome := range chromosomes {
		chromosomeDirectory := filepath.Join(directory, fmt.Sprintf("chr%d", chromosome))
		genePath := filepath.Join(chromosomeDirectory, "genes.txt")
		genes, keysA, err := readGenes(genePath, filepath.Join(chromosomeDirectory, "A", "keys.tsv"))
		if err != nil {
			return nil, err
		}
		_, keysB, err := readGenes(genePath, filepath.Join(chromosomeDirectory, "B", "keys.tsv"))
		if err != nil {
			return nil, err
		}
		for i, gene := range genes {
			inA := make(map[string]bool, len(keysA[i]))
			for _, key := range keysA[i] {
				inA[key] = true
			}
			inB := make(map[string]bool, len(keysB[i]))
			for _, key := range keysB[i] {
				inB[key] = true
			}
			both := 0
			for key := range inB {
				if inA[key] {
					both++
				}
			}
			empty := cap.EmptyBuckets(cap.LocalBitmap(hp, capKey, append(slices.Clone(keysA[i]), keysB[i]...)))
			stats = append(stats, capStat{
				chromosome: chromosome,
				geneID:     gene.GeneID,
				aOnly:      len(inA) - both,
				bOnly:      len(inB) - both,
				both:       both,
				union:      len(inA) + len(inB) - both,
				capM:       th.Ladder[cap.SelectIndex(th, empty)],
			})
		}
	}
	return stats, nil
}

// printCapStats writes the per-chromosome totals and the per-chromosome, per-cap union statistics.
func printCapStats(w io.Writer, stats []capStat, chromosomes []int) {
	byChromosome := map[int][]capStat{}
	for _, s := range stats {
		byChromosome[s.chromosome] = append(byChromosome[s.chromosome], s)
	}

	fmt.Fprintln(w, "\n[1] Per chromosome: variants per gene, summed over genes")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "chr\tgenes\tunion=0\tA-only\tB-only\tboth\tunion\tmax union\t")
	row := func(label string, group []capStat) {
		var aOnly, bOnly, both, union, maxUnion, empty int
		for _, s := range group {
			aOnly += s.aOnly
			bOnly += s.bOnly
			both += s.both
			union += s.union
			maxUnion = max(maxUnion, s.union)
			if s.union == 0 {
				empty++
			}
		}
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t\n", label, len(group), empty, aOnly, bOnly, both, union, maxUnion)
	}
	for _, chromosome := range chromosomes {
		row(fmt.Sprintf("chr%d", chromosome), byChromosome[chromosome])
	}
	row("ALL", stats)
	tw.Flush()

	fmt.Fprintln(w, "\n[2] Per chromosome and cap M: real union size of the genes that got that cap")
	fmt.Fprintln(w, "    union/M is the fill ratio (at most 1 when the cap holds); M/union median is the slack.")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "chr\tcap M\tgenes\tmin\tmedian\tmax\tmax union/M\tmedian M/union\tunion>M\t")
	capRows := func(label string, group []capStat) {
		byCap := map[int][]capStat{}
		var caps []int
		for _, s := range group {
			if _, ok := byCap[s.capM]; !ok {
				caps = append(caps, s.capM)
			}
			byCap[s.capM] = append(byCap[s.capM], s)
		}
		slices.Sort(caps)
		for _, capM := range caps {
			genes := byCap[capM]
			unions := make([]int, len(genes))
			slack := make([]float64, 0, len(genes))
			over, maxFill := 0, 0.0
			for i, s := range genes {
				unions[i] = s.union
				maxFill = max(maxFill, float64(s.union)/float64(capM))
				if s.union > capM {
					over++
				}
				if s.union > 0 {
					slack = append(slack, float64(capM)/float64(s.union))
				}
			}
			slices.Sort(unions)
			slices.Sort(slack)
			medianSlack := "-"
			if len(slack) > 0 {
				medianSlack = fmt.Sprintf("%.2f", slack[len(slack)/2])
			}
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%d\t%.3f\t%s\t%d\t\n",
				label, capM, len(genes), unions[0], unions[len(unions)/2], unions[len(unions)-1], maxFill, medianSlack, over)
		}
	}
	for _, chromosome := range chromosomes {
		capRows(fmt.Sprintf("chr%d", chromosome), byChromosome[chromosome])
	}
	capRows("ALL", stats)
	tw.Flush()
}
