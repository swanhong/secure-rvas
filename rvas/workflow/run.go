package workflow

import (
	"fmt"
	"path/filepath"
	"runtime"
	"time"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/hhcho/sfgwas/rvas/protocol"
	"gonum.org/v1/gonum/mat"
)

type chromosomeResult struct {
	Chromosome int
	Genes      []protocol.Gene
	BurdenP    []float64
	SKATWHP    []float64
}

type nullModel struct {
	beta   mpc_core.RMat
	xtxInv mpc_core.RMat
	rss    mpc_core.RVec
}

func runChromosome(
	secureSession *session,
	model nullModel,
	input *PartyInput,
	chromosomeIndex int,
	seed int64,
) (chromosomeResult, error) {
	chromosome := input.Chromosomes[chromosomeIndex]
	dataParams := secureSession.dataParams[chromosomeIndex]
	networks := mpc.ParallelNetworks(secureSession.networks)
	laneObservers := make([]func(stage string) func(), len(networks))
	for lane, network := range networks {
		laneObservers[lane] = secureSession.metrics.observe(
			chromosome.Chromosome,
			mpc.ParallelNetworks{network},
		)
	}
	observe := laneObservers[0]
	observeBatch := secureSession.metrics.observePackedWidth(
		chromosome.Chromosome,
		chromosome.CryptoParams.R,
	)
	mpcObject := secureSession.mpcObjects[0]
	embeddings := secureSession.alignments[chromosomeIndex].E

	var localDosage *mat.Dense
	if chromosome.GenotypeDirectory != "" {
		var err error
		localDosage, err = readDosageSums(
			chromosome.GenotypeDirectory,
			chromosome.Genes,
			chromosome.Keys,
			input.SampleCount,
			observe,
		)
		if err != nil {
			return chromosomeResult{}, err
		}
	}

	done := secureSession.metrics.start(
		"compute_weights",
		chromosome.Chromosome,
		networks,
	)
	weight, signedWeight := protocol.ComputeWeights(
		mpcObject,
		secureSession.heContext,
		dataParams,
		localDosage,
		embeddings,
	)
	done()

	loadGenotypes := func(
		batch protocol.GeneBatch,
	) ([]*mat.Dense, error) {
		return readGeneBatch(
			chromosome.GenotypeDirectory,
			chromosome.Genes,
			chromosome.Keys,
			batch,
			input.SampleCount,
		)
	}

	done = secureSession.metrics.start(
		"packed_statistics",
		chromosome.Chromosome,
		networks,
	)
	geneQ, geneL, geneV, geneInvS1, geneS2, geneS3, err :=
		protocol.ComputePackedStatistics(
			secureSession.mpcObjects,
			secureSession.heContext,
			dataParams,
			chromosome.CryptoParams,
			loadGenotypes,
			input.X,
			input.Y,
			model.beta,
			model.xtxInv,
			weight,
			signedWeight,
			embeddings,
			seed,
			laneObservers,
			observeBatch,
		)
	done()
	if err != nil {
		return chromosomeResult{}, err
	}

	done = secureSession.metrics.start(
		"finalize",
		chromosome.Chromosome,
		networks,
	)
	bSquared, z := protocol.Finalize(
		secureSession.mpcObjects,
		dataParams,
		geneQ,
		geneL,
		model.rss,
		geneV,
		geneInvS1,
		geneS2,
		geneS3,
		observe,
	)
	done()

	done = secureSession.metrics.start(
		"release",
		chromosome.Chromosome,
		networks,
	)
	burdenP, skatWHP := protocol.Release(
		mpcObject,
		bSquared,
		z,
		observe,
	)
	done()

	return chromosomeResult{
		Chromosome: chromosome.Chromosome,
		Genes:      dataParams.Genes,
		BurdenP:    burdenP,
		SKATWHP:    skatWHP,
	}, nil
}

// runParty loads every ancestry's input before any computation, so a bad or missing input file
// fails in seconds instead of after the earlier ancestries have run for hours.
func runParty(config *Config, partyID int) error {
	inputs := make([]*PartyInput, len(config.Ancestries))
	loadDurations := make([]time.Duration, len(config.Ancestries))
	for index, ancestry := range config.Ancestries {
		startedAt := time.Now()
		input, err := loadPartyInput(config, partyID, ancestry)
		if err != nil {
			return fmt.Errorf("load party %d input for ancestry %s: %w", partyID, ancestry, err)
		}
		if err := loadEmbeddingCaps(config.Embedding.CapsDir, ancestry, partyID, input); err != nil {
			return err
		}
		inputs[index], loadDurations[index] = input, time.Since(startedAt)
	}
	for index, ancestry := range config.Ancestries {
		if err := runAncestry(
			config,
			partyID,
			filepath.Join(config.SharedKeysPath, ancestry),
			ancestry,
			inputs[index],
			loadDurations[index],
		); err != nil {
			return fmt.Errorf("run ancestry %s: %w", ancestry, err)
		}
		inputs[index] = nil
	}
	return nil
}

func runAncestry(
	config *Config,
	partyID int,
	sharedKeysPath string,
	ancestry string,
	input *PartyInput,
	inputLoading time.Duration,
) error {
	metrics := newMetricRecorder(
		fmt.Sprintf("party%d", partyID),
		ancestry,
		partyID == cohortAPartyID,
	)
	metrics.addDuration("input_loading", 0, inputLoading)

	secureSession, err := openSession(
		config,
		partyID,
		input,
		sharedKeysPath,
		metrics,
	)
	if err != nil {
		return fmt.Errorf(
			"open party %d session: %w",
			partyID,
			err,
		)
	}
	defer secureSession.close()
	if partyID == cohortAPartyID {
		if err := writeEmbeddingPlans(config.RunDir, ancestry, input, secureSession.alignments); err != nil {
			return err
		}
	}

	networks := mpc.ParallelNetworks(secureSession.networks)
	done := metrics.start("null_model", 0, networks)
	beta, xtxInv, rss := protocol.SetupNull(
		secureSession.mpcObjects[0],
		secureSession.dataParams[0],
		input.X,
		input.Y,
		metrics.observe(0, networks),
	)
	done()
	model := nullModel{
		beta:   beta,
		xtxInv: xtxInv,
		rss:    rss,
	}

	results := make(
		[]chromosomeResult,
		0,
		len(input.Chromosomes),
	)
	for chromosomeIndex := range input.Chromosomes {
		chromosome := input.Chromosomes[chromosomeIndex].Chromosome
		done := metrics.start(
			"chromosome_total",
			chromosome,
			networks,
		)
		result, err := runChromosome(
			secureSession,
			model,
			input,
			chromosomeIndex,
			config.Seed,
		)
		done()
		if err != nil {
			return fmt.Errorf("run chromosome %d: %w", chromosome, err)
		}
		if partyID == cohortAPartyID {
			results = append(results, result)
			if err := writeChromosomeResult(config.RunDir, ancestry, result, config.PhenotypeColumns); err != nil {
				return fmt.Errorf("write chromosome %d results: %w", chromosome, err)
			}
		}
		runtime.GC()
	}

	for _, m := range secureSession.mpcObjects {
		m.AssertSync()
	}

	var resultErr error
	if partyID == cohortAPartyID {
		done := metrics.start("write_results", 0, nil)
		resultErr = writeSecureResults(
			config.RunDir,
			ancestry,
			results,
			config.PhenotypeColumns,
		)
		done()
	}

	metricsPath := filepath.Join(
		config.RunDir,
		"metrics",
		ancestry,
		fmt.Sprintf("metrics_party%d.csv", partyID),
	)
	metricsErr := metrics.writeCSV(metricsPath)
	if resultErr != nil {
		return resultErr
	}
	if metricsErr != nil {
		return fmt.Errorf("write party %d metrics: %w", partyID, metricsErr)
	}
	metrics.printTimeTree()
	return nil
}
