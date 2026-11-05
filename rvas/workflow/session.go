package workflow

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strconv"

	mpc_core "github.com/hhcho/mpc-core"
	securecrypto "github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/hhcho/sfgwas/rvas/embed"
	"github.com/hhcho/sfgwas/rvas/protocol"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	mpcFieldBits     = 256
	divSqrtMaxLength = 64
)

type session struct {
	networks   []*mpc.Network
	heContext  *securecrypto.CryptoParams
	mpcObjects []*mpc.MPC
	metrics    *metricRecorder
	dataParams []protocol.DataParams
	alignments []embed.PipelineResult // per chromosome: caps and embeddings
}

// Fixed-size agreement precedes every variable-size protocol.
func exchangePublicParameters(network *mpc.Network, config *Config, partyID, localSampleCount int, chromosomes []ChromosomeInput) (int, error) {
	embedding := config.Embedding
	embedding.CapsDir = "" // Local path; RunEmbedding agrees on the cap values.
	public, err := json.Marshal(struct {
		CKKS                                            string
		Threads, DataBits, FracBits, Probes, Covariates int
		Seed                                            int64
		Chromosomes                                     []int
		Ancestries, Phenotypes                          []string
		Embedding                                       EmbeddingConfig
	}{config.CKKS, config.MpcNumThreads, config.DataBits, config.FractionalBits,
		config.Probes, config.NumCov, config.Seed, config.Chromosomes,
		config.Ancestries, config.PhenotypeColumns, embedding})
	message := make([]uint64, 10)
	if err == nil && (partyID == 0 || localSampleCount > 0) {
		message[0] = 1
	}
	message[1] = uint64(localSampleCount)
	digest := sha256.Sum256(public)
	for i := 0; i < 4; i++ {
		message[2+i] = binary.BigEndian.Uint64(digest[8*i:])
	}
	copy(message[6:], geneOrderDigest(chromosomes))
	reply := make([]uint64, 2)
	if partyID == 0 {
		reply[0] = message[0]
		var genes []uint64
		for site := 1; site < network.NumParties; site++ {
			other := network.ReceiveIntVector(10, site)
			if site == 1 {
				genes = other[6:]
			}
			if other[0] != 1 || !slices.Equal(message[2:6], other[2:6]) || !slices.Equal(genes, other[6:]) {
				reply[0] = 0
			}
			reply[1] += other[1]
		}
		for site := 1; site < network.NumParties; site++ {
			network.SendIntVector(reply, site)
		}
	} else {
		network.SendIntVector(message, 0)
		reply = network.ReceiveIntVector(2, 0)
	}
	if reply[0] != 1 {
		return 0, fmt.Errorf("public protocol configuration or gene order mismatch, or invalid sample count")
	}
	if partyID == 1 {
		counts := make([]uint64, len(chromosomes))
		for i := range chromosomes {
			counts[i] = uint64(len(chromosomes[i].Genes))
		}
		network.SendIntVector(counts, 0)
	} else if partyID == 0 {
		counts := network.ReceiveIntVector(len(chromosomes), 1)
		for i := range chromosomes {
			chromosomes[i].Genes = make([]protocol.Gene, counts[i])
		}
	}
	return int(reply[1]), nil
}

// geneOrderDigest is SHA-256 over the chromosomes and gene IDs in order, as four words.
func geneOrderDigest(chromosomes []ChromosomeInput) []uint64 {
	hash := sha256.New()
	for _, chromosome := range chromosomes {
		fmt.Fprintf(hash, "chr%d\n", chromosome.Chromosome)
		for _, gene := range chromosome.Genes {
			fmt.Fprintln(hash, gene.GeneID)
		}
	}
	sum := hash.Sum(nil)
	digest := make([]uint64, len(sum)/8)
	for i := range digest {
		digest[i] = binary.BigEndian.Uint64(sum[8*i:])
	}
	return digest
}

func openSession(
	config *Config,
	partyID int,
	input *PartyInput,
	sharedKeysPath string,
	metrics *metricRecorder,
) (*session, error) {
	if sharedKeysPath == "" {
		return nil, fmt.Errorf("shared keys path is required")
	}

	literal, err := securecrypto.ResolveCKKSParametersLiteral(config.CKKS)
	if err != nil {
		return nil, err
	}
	heParameters, err := ckks.NewParametersFromLiteral(literal)
	if err != nil {
		return nil, err
	}
	done := metrics.start("network_init", 0, nil)
	networks := mpc.InitCommunication(
		config.BindingIP,
		config.Servers,
		partyID,
		len(config.Servers),
		config.MpcNumThreads,
		sharedKeysPath,
	)
	done()

	parallelNetworks := mpc.ParallelNetworks(networks)
	for _, network := range networks {
		network.EnableLogging()
	}

	done = metrics.start(
		"sample_count_exchange",
		0,
		parallelNetworks,
	)
	totalSamples, err := exchangePublicParameters(
		networks[0],
		config,
		partyID,
		input.SampleCount,
		input.Chromosomes,
	)
	done()
	if err != nil {
		for _, network := range networks {
			network.CloseAll()
		}
		return nil, err
	}

	mpcObjects := mpc.InitParallelMPCEnv(
		networks,
		mpc_core.LElem256Zero,
		config.DataBits,
		config.FractionalBits,
	)
	booleanShares := mpcObjects[0].GetBooleanShareFlag()
	for _, mpcObject := range mpcObjects {
		mpcObject.SetHubPid(cohortAPartyID)
		mpcObject.SetBooleanShareFlag(booleanShares)
		mpcObject.SetDivSqrtMaxLen(divSqrtMaxLength)
	}

	// Pack genes across chromosomes by cap, with one embedding key context.
	done = metrics.start("alignment", 0, parallelNetworks)
	endPhase := func() {}
	alignments, err := runAlignment(mpcObjects[0], config, input.Chromosomes, embed.PipelineObserver{
		Phase: func(phase string) {
			endPhase()
			endPhase = func() {}
			if phase != "" {
				endPhase = metrics.start("embedding_"+phase, 0, mpc.ParallelNetworks{networks[0]})
			}
		},
		Caps: func(caps []int) {
			if metrics.printEvents {
				fmt.Printf("\n[%s %s] gene counts by cap and chromosome:\n%s\n", metrics.process, metrics.ancestry, capCountTable(input.Chromosomes, caps))
			}
		},
		CoordinateStep: func(m int, stage string) func() {
			return metrics.start(fmt.Sprintf("embedding_M%d_coordinates_%s", m, stage), 0, mpc.ParallelNetworks{networks[0]})
		},
	})
	done()
	if err != nil {
		for _, network := range networks {
			network.CloseAll()
		}
		return nil, err
	}
	mpcObjects[0].AssertSync()
	for index := range input.Chromosomes {
		for geneIndex, capM := range alignments[index].Caps {
			input.Chromosomes[index].Genes[geneIndex].Cap = capM
		}
	}

	for index, chromosome := range input.Chromosomes {
		dataParams := protocol.DataParams{
			Genes:          chromosome.Genes,
			C:              config.NumCov + 1,
			PhenotypeCount: len(config.PhenotypeColumns),
		}
		cryptoParams, err := protocol.BuildCryptoParams(
			dataParams,
			config.Probes,
			heParameters.MaxSlots(),
		)
		if err != nil {
			for _, network := range networks {
				network.CloseAll()
			}
			return nil, err
		}
		input.Chromosomes[index].CryptoParams = cryptoParams
	}
	galoisElements := requiredGaloisElements(
		heParameters,
		input.Chromosomes,
	)

	dataParams := make(
		[]protocol.DataParams,
		len(input.Chromosomes),
	)
	for index, chromosome := range input.Chromosomes {
		dataParams[index] = protocol.DataParams{
			Genes:          chromosome.Genes,
			N:              totalSamples,
			C:              config.NumCov + 1,
			PhenotypeCount: len(config.PhenotypeColumns),
		}
	}

	done = metrics.start("collective_setup", 0, parallelNetworks)
	heContext := parallelNetworks.CollectiveInit(
		&heParameters,
		mpcFieldBits,
		true,
		galoisElements,
	)
	metrics.addDuration("pubkey_gen", 0, mpc.SetupTiming.PubKey)
	metrics.addDuration("relin_key_gen", 0, mpc.SetupTiming.RelinKey)
	metrics.addDuration("rotkey_gen", 0, mpc.SetupTiming.RotKey)
	done()

	return &session{
		networks:   networks,
		heContext:  heContext,
		mpcObjects: mpcObjects,
		metrics:    metrics,
		dataParams: dataParams,
		alignments: alignments,
	}, nil
}

func (session *session) close() {
	for _, network := range session.networks {
		network.CloseAll()
	}
}

func requiredGaloisElements(
	heParameters ckks.Parameters,
	chromosomes []ChromosomeInput,
) []uint64 {
	unique := make(map[uint64]struct{})

	for _, chromosome := range chromosomes {
		cryptoParams := chromosome.CryptoParams

		for _, batch := range cryptoParams.Batches {
			nu := cryptoParams.Slots / batch.W
			for _, element := range heParameters.GaloisElementsForInnerSum(
				nu,
				batch.W,
			) {
				unique[element] = struct{}{}
			}
		}
	}

	elements := make([]uint64, 0, len(unique))
	for element := range unique {
		elements = append(elements, element)
	}
	sort.Slice(elements, func(left, right int) bool {
		return elements[left] < elements[right]
	})

	return elements
}

func runAlignment(m *mpc.MPC, config *Config, chromosomes []ChromosomeInput, observer embed.PipelineObserver) ([]embed.PipelineResult, error) {
	// The actual gene order was agreed before this call. Use public indices so
	// P0 does not need gene names, and equal names on different chromosomes stay distinct.
	var ids []string
	var keys [][]string
	var caps []int
	for _, chromosome := range chromosomes {
		for range chromosome.Genes {
			ids = append(ids, strconv.Itoa(len(ids)))
		}
		keys = append(keys, chromosome.Keys...)
		if chromosome.PublicCaps != nil {
			if caps == nil {
				caps = []int{}
			}
			caps = append(caps, chromosome.PublicCaps...)
		}
	}
	result, err := embed.RunEmbeddingObserved(m, embed.PipelineConfig{
		Genes: ids, Cap: config.Embedding.Cap, CKKS: config.CKKS, Mode: config.Embedding.Mode, DenseMax: config.Embedding.DenseMax, Hash: config.Embedding.Hash, Caps: caps,
	}, keys, observer)
	if err != nil {
		return nil, err
	}
	return splitAlignment(chromosomes, result), nil
}

// Slice E shares without copying matrices; downstream statistics retain chromosome-local indices.
func splitAlignment(chromosomes []ChromosomeInput, result embed.PipelineResult) []embed.PipelineResult {
	results := make([]embed.PipelineResult, len(chromosomes))
	first := 0
	for i, chromosome := range chromosomes {
		end := first + len(chromosome.Genes)
		local := &results[i]
		local.Caps = result.Caps[first:end]
		local.Galois, local.GaloisKeyBytes = result.Galois, result.GaloisKeyBytes
		local.E = make([][]mpc_core.RMat, len(result.E))
		for site := range result.E {
			local.E[site] = result.E[site][first:end]
		}
		for _, batch := range result.Batches {
			var genes []int
			for _, g := range batch.Genes {
				if first <= g && g < end {
					genes = append(genes, g-first)
				}
			}
			if len(genes) > 0 {
				local.Batches = append(local.Batches, embed.EmbeddingBatch{Setting: batch.Setting, Genes: genes})
			}
		}
		first = end
	}
	return results
}
