package mpc

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/multiparty/mpckks"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
	"go.dedis.ch/onet/v3/log"
)

func (netObj *Network) sharedCRS() sampling.PRNG {
	seed := make([]byte, 64)
	netObj.Rand.SwitchPRG(-1)
	netObj.Rand.RandRead(seed)
	netObj.Rand.RestorePRG()

	crs, err := sampling.NewKeyedPRNG(seed)
	if err != nil {
		panic(err)
	}

	return crs
}

func (netObj *Network) aggregateRelinearizationKeyGenShare(prot multiparty.RelinearizationKeyGenProtocol, share multiparty.RelinearizationKeyGenShare) multiparty.RelinearizationKeyGenShare {
	_, outRoundOne, outRoundTwo := prot.AllocateShare()
	out := outRoundOne
	if share.Degree() == 0 {
		out = outRoundTwo
	}

	pid := netObj.GetPid()
	hubPid := netObj.GetHubPid()

	if pid == 0 {
		if hubPid > 0 {
			if err := out.UnmarshalBinary(netObj.receiveBinaryValue(hubPid)); err != nil {
				panic(err)
			}
		}
		return out
	}

	if pid == hubPid {
		for p := 1; p < netObj.GetNParty(); p++ {
			var next multiparty.RelinearizationKeyGenShare
			if p == pid {
				next = share
			} else {
				_, nextRoundOne, nextRoundTwo := prot.AllocateShare()
				next = nextRoundOne
				if share.Degree() == 0 {
					next = nextRoundTwo
				}
				if err := next.UnmarshalBinary(netObj.receiveBinaryValue(p)); err != nil {
					panic(err)
				}
			}
			prot.AggregateShares(next, out, &out)
		}

		for p := 0; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.sendBinaryValue(out, p)
			}
		}
		return out
	}

	netObj.sendBinaryValue(share, hubPid)
	if err := out.UnmarshalBinary(netObj.receiveBinaryValue(hubPid)); err != nil {
		panic(err)
	}
	return out
}

func (netObj *Network) aggregateKeySwitchShare(prot multiparty.KeySwitchProtocol, share multiparty.KeySwitchShare, level int) multiparty.KeySwitchShare {
	out := prot.AllocateShare(level)
	pid := netObj.GetPid()
	hubPid := netObj.GetHubPid()

	if pid == hubPid {
		for p := 1; p < netObj.GetNParty(); p++ {
			next := prot.AllocateShare(level)
			if p == pid {
				next = share
			} else if err := next.UnmarshalBinary(netObj.receiveBinaryValue(p)); err != nil {
				panic(err)
			}
			if err := prot.AggregateShares(next, out, &out); err != nil {
				panic(err)
			}
		}

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.sendBinaryValue(out, p)
			}
		}
		return out
	}

	netObj.sendBinaryValue(share, hubPid)
	if err := out.UnmarshalBinary(netObj.receiveBinaryValue(hubPid)); err != nil {
		panic(err)
	}
	return out
}

// SetupTiming records the collective crypto-setup phase durations (for the skat_fed timing tree).
// It describes the last completed setup in a process. Concurrent local-harness
// parties use their own timers rather than reading this shared summary.
var SetupTiming struct{ PubKey, RelinKey, RotKey time.Duration }
var setupTimingMu sync.Mutex

func (netObj ParallelNetworks) CollectiveInit(params *ckks.Parameters, prec uint, rotPow2Only bool, galoisElements []uint64) (cps *crypto.CryptoParams) {
	log.LLvl1("CollectiveInit started")
	var timing struct{ PubKey, RelinKey, RotKey time.Duration }
	defer func() {
		setupTimingMu.Lock()
		SetupTiming = timing
		setupTimingMu.Unlock()
	}()

	kgen := ckks.NewKeyGenerator(*params)

	skShard := rlwe.NewSecretKey(*params)
	if netObj[0].GetPid() > 0 {
		skShard = kgen.GenSecretKeyNew()
	}

	baseCRS := netObj[0].sharedCRS()

	t := time.Now()
	pk := netObj[0].CollectivePubKeyGen(params, skShard, baseCRS)
	timing.PubKey = time.Since(t)
	log.LLvl1(fmt.Sprintf("PubKeyGen %v", timing.PubKey.Round(time.Millisecond)))

	t = time.Now()
	rlk := netObj[0].CollectiveRelinKeyGen(params, skShard, baseCRS)
	timing.RelinKey = time.Since(t)
	log.LLvl1(fmt.Sprintf("RelinKeyGen %v", timing.RelinKey.Round(time.Millisecond)))

	nprocs := runtime.GOMAXPROCS(0)
	cps = crypto.NewCryptoParams(*params, skShard, skShard.CopyNew(), pk, rlk, prec, nprocs)

	smallDim := 20
	babyFlag := true
	if rotPow2Only { // config rotkey_pow2only → power-of-two rotation keys only (InnerSumAll)
		smallDim, babyFlag = 0, false
	}
	if galoisElements == nil {
		log.LLvl1("RotKeyGen: shifts <=", smallDim, "babyFlag", babyFlag, "powers of two up to", cps.GetSlots())
	} else {
		log.LLvl1("RotKeyGen: manifest Galois elements", len(galoisElements))
	}
	if strings.TrimSpace(os.Getenv("SFGWAS_SKIP_ROTKEYGEN")) != "" {
		if len(galoisElements) > 0 {
			panic("CollectiveInit: cannot skip manifest rotation keys")
		}
		log.LLvl1("CollectiveInit: skipping rotation-key generation because SFGWAS_SKIP_ROTKEYGEN is set")
		log.LLvl1("CollectiveInit finished")
		return
	}

	t = time.Now()
	if netObj[0].GetPid() > 0 {
		crsVec := make([]sampling.PRNG, len(netObj))
		for i := range netObj {
			crsVec[i] = netObj[i].sharedCRS()
		}
		var rotKs []*rlwe.GaloisKey
		if galoisElements == nil {
			rotKs = netObj.CollectiveRotKeyGen(params, skShard, crsVec, crypto.GenerateRotKeys(cps.GetSlots(), smallDim, babyFlag))
		} else {
			rotKs = netObj.CollectiveGaloisKeyGen(params, skShard, crsVec, galoisElements)
		}
		cps.SetEvaluators(*params, rlk, rotKs)
	}
	timing.RotKey = time.Since(t)
	log.LLvl1(fmt.Sprintf("CollectiveInit finished (RotKeyGen %v) | setup total %v",
		timing.RotKey.Round(time.Millisecond), (timing.PubKey + timing.RelinKey + timing.RotKey).Round(time.Millisecond)))
	return
}

func (netObj *Network) CollectivePubKeyGen(parameters *ckks.Parameters, skShard *rlwe.SecretKey, crs sampling.PRNG) (pk *rlwe.PublicKey) {
	ckgProtocol := multiparty.NewPublicKeyGenProtocol(*parameters)
	pkShare := ckgProtocol.AllocateShare()
	crp := ckgProtocol.SampleCRP(crs)

	if netObj.GetPid() > 0 {
		ckgProtocol.GenShare(skShard, crp, &pkShare)
	}

	pkAgg := ckgProtocol.AllocateShare()
	pid := netObj.GetPid()
	hubPid := netObj.GetHubPid()

	if pid == 0 {
		if hubPid > 0 {
			if err := pkAgg.UnmarshalBinary(netObj.receiveBinaryValue(hubPid)); err != nil {
				panic(err)
			}
		}
	} else if pid == hubPid {
		for p := 1; p < netObj.GetNParty(); p++ {
			next := ckgProtocol.AllocateShare()
			if p == pid {
				next = pkShare
			} else if err := next.UnmarshalBinary(netObj.receiveBinaryValue(p)); err != nil {
				panic(err)
			}
			ckgProtocol.AggregateShares(next, pkAgg, &pkAgg)
		}

		for p := 0; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.sendBinaryValue(pkAgg, p)
			}
		}
	} else {
		netObj.sendBinaryValue(pkShare, hubPid)
		if err := pkAgg.UnmarshalBinary(netObj.receiveBinaryValue(hubPid)); err != nil {
			panic(err)
		}
	}

	pk = rlwe.NewPublicKey(*parameters)
	ckgProtocol.GenPublicKey(pkAgg, crp, pk)
	return
}

var decryptFlooding = ring.DiscreteGaussian{Sigma: 6.36, Bound: 38}

func (netObj *Network) CollectiveDecryptMat(cps *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid int) (pm crypto.PlainMatrix) {
	pid := netObj.GetPid()
	if pid == 0 {
		return
	}

	var tmp crypto.CipherMatrix
	var nr, nc int

	if sourcePid > 0 {
		if pid == sourcePid {
			nr, nc = len(cm), len(cm[0])
			for p := 1; p < netObj.GetNParty(); p++ {
				if p != sourcePid {
					netObj.SendInt(nr, p)
					netObj.SendInt(nc, p)
				}
			}
		} else {
			nr = netObj.ReceiveInt(sourcePid)
			nc = netObj.ReceiveInt(sourcePid)
		}

		tmp = netObj.BroadcastCMat(cps, cm, sourcePid, nr, nc)
	} else {
		nr = len(cm)
		nc = len(cm[0])
		tmp = crypto.CopyEncryptedMatrix(cm)
	}

	tmp, level := crypto.FlattenLevels(cps, tmp)

	ksp, err := multiparty.NewKeySwitchProtocol(cps.Params, decryptFlooding)
	if err != nil {
		panic(err)
	}

	zero := rlwe.NewSecretKey(cps.Params)
	pm = make(crypto.PlainMatrix, nr)
	for i := range pm {
		pm[i] = make(crypto.PlainVector, nc)
		for j := range pm[i] {
			decShare := ksp.AllocateShare(level)
			ksp.GenShare(cps.Sk, zero, tmp[i][j], &decShare)
			decAgg := netObj.aggregateKeySwitchShare(ksp, decShare, level)

			ciphertextSwitched := rlwe.NewCiphertext(cps.Params, 1, level)
			ksp.KeySwitch(tmp[i][j], decAgg, ciphertextSwitched)
			pm[i][j] = ciphertextSwitched.Plaintext()
		}
	}

	return
}

func (netObj *Network) CollectiveDecryptVec(cps *crypto.CryptoParams, cv crypto.CipherVector, sourcePid int) (pv crypto.PlainVector) {
	if netObj.GetPid() == 0 {
		return
	}
	return netObj.CollectiveDecryptMat(cps, crypto.CipherMatrix{cv}, sourcePid)[0]
}

func (netObj *Network) CollectiveDecrypt(cps *crypto.CryptoParams, ct *rlwe.Ciphertext, sourcePid int) (pt *rlwe.Plaintext) {
	if netObj.GetPid() == 0 {
		return
	}

	tmp := ct
	if sourcePid > 0 {
		tmp = netObj.BroadcastCiphertext(cps, ct, sourcePid)
	}

	ksp, err := multiparty.NewKeySwitchProtocol(cps.Params, decryptFlooding)
	if err != nil {
		panic(err)
	}

	decShare := ksp.AllocateShare(tmp.Level())
	zero := rlwe.NewSecretKey(cps.Params)
	ksp.GenShare(cps.Sk, zero, tmp, &decShare)
	decAgg := netObj.aggregateKeySwitchShare(ksp, decShare, tmp.Level())

	ciphertextSwitched := rlwe.NewCiphertext(cps.Params, 1, tmp.Level())
	ksp.KeySwitch(tmp, decAgg, ciphertextSwitched)
	return ciphertextSwitched.Plaintext()
}

func (netObj *Network) CollectiveBootstrapVec(cps *crypto.CryptoParams, cv crypto.CipherVector, sourcePid int) crypto.CipherVector {
	return netObj.CollectiveBootstrapMat(cps, crypto.CipherMatrix{cv}, sourcePid)[0]
}

func (netObj *Network) CanCollectiveBootstrap(cps *crypto.CryptoParams, level int) bool {
	minLevel, _, ok := mpckks.GetMinimumLevelForRefresh(128, cps.Params.DefaultScale(), max(1, netObj.GetNParty()-1), cps.Params.Q())
	return ok && level >= minLevel
}

func (netObj *Network) CollectiveBootstrapMat(cps *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid int) crypto.CipherMatrix {
	if netObj.GetPid() == 0 {
		return cm
	}

	if sourcePid > 0 {
		if netObj.GetPid() == sourcePid {
			cm = crypto.RerandomizeMatrix(cps, cm)
			for p := 1; p < netObj.GetNParty(); p++ {
				if p != sourcePid {
					netObj.SendInt(len(cm), p)
					netObj.SendInt(len(cm[0]), p)
					netObj.SendCipherMatrix(cm, p)
				}
			}
		} else {
			nrows := netObj.ReceiveInt(sourcePid)
			ncols := netObj.ReceiveInt(sourcePid)
			cm = netObj.ReceiveCipherMatrix(cps, nrows, ncols, sourcePid)
		}
	}

	cm, _ = crypto.FlattenLevels(cps, cm)

	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, cps.Params.DefaultScale(), max(1, netObj.GetNParty()-1), cps.Params.Q())
	if !ok || cm[0][0].Level() < minLevel {
		panic(fmt.Sprintf("CollectiveBootstrapMat: ciphertext level %d below required refresh level %d", cm[0][0].Level(), minLevel))
	}

	rfp, err := mpckks.NewRefreshProtocol(cps.Params, cps.GetPrec(), cps.Params.Xe())
	if err != nil {
		panic(err)
	}

	crs := netObj.sharedCRS()
	for i := range cm {
		for j := range cm[i] {
			crp := rfp.SampleCRP(cps.Params.MaxLevel(), crs)
			share := rfp.AllocateShare(cm[i][j].Level(), cps.Params.MaxLevel())
			if err := rfp.GenShare(cps.Sk, logBound, cm[i][j], crp, &share); err != nil {
				panic(err)
			}

			agg := rfp.AllocateShare(cm[i][j].Level(), cps.Params.MaxLevel())
			if netObj.GetPid() == netObj.GetHubPid() {
				for p := 1; p < netObj.GetNParty(); p++ {
					next := rfp.AllocateShare(cm[i][j].Level(), cps.Params.MaxLevel())
					if p == netObj.GetPid() {
						next = share
					} else if err := next.UnmarshalBinary(netObj.receiveBinaryValue(p)); err != nil {
						panic(err)
					}
					if err := rfp.AggregateShares(&next, &agg, &agg); err != nil {
						panic(err)
					}
				}

				for p := 1; p < netObj.GetNParty(); p++ {
					if p != netObj.GetPid() {
						netObj.sendBinaryValue(agg, p)
					}
				}
			} else {
				netObj.sendBinaryValue(share, netObj.GetHubPid())
				if err := agg.UnmarshalBinary(netObj.receiveBinaryValue(netObj.GetHubPid())); err != nil {
					panic(err)
				}
			}
			agg.MetaData = *cm[i][j].MetaData

			out := cm[i][j].CopyNew()
			if err := rfp.Finalize(cm[i][j], crp, agg, out); err != nil {
				panic(err)
			}
			cm[i][j] = out
		}
	}

	return cm
}

// RefreshCMatWithRandomZero transfers a source's live ciphertexts once, then refreshes
// them in public row-major order. Each entry retains its own input level until
// refresh; the matrix is not flattened to its lowest level. All data sites get
// the output, so a subsequent CMatToSS uses sourcePid=-1 to avoid retransmission.
// The existing refresh masks, admission rule, and hub protocol are unchanged.
func (netObj *Network) RefreshCMatWithRandomZero(cp *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid, rows, cols int) crypto.CipherMatrix {
	cm = netObj.BroadcastCMatWithRandomZero(cp, cm, sourcePid, rows, cols)
	if netObj.GetPid() == 0 {
		return cm
	}
	for i := range cm {
		for j, ct := range cm[i] {
			cm[i][j] = netObj.CollectiveBootstrapMat(cp, crypto.CipherMatrix{{ct}}, -1)[0][0]
		}
	}
	return cm
}

// BootstrapMatAll: collective bootstrap for all parties (except 0)
func (netObj *Network) BootstrapMatAll(cps *crypto.CryptoParams, cm crypto.CipherMatrix) crypto.CipherMatrix {
	tmp := make(crypto.CipherMatrix, len(cm))

	for sourcePid := 1; sourcePid < netObj.GetNParty(); sourcePid++ {
		if netObj.GetPid() == sourcePid {
			cm = netObj.CollectiveBootstrapMat(cps, cm, sourcePid)
		} else {
			netObj.CollectiveBootstrapMat(cps, tmp, sourcePid)
		}
	}

	return cm
}

func (netObj *Network) BootstrapVecAll(cps *crypto.CryptoParams, cv crypto.CipherVector) crypto.CipherVector {
	tmp := make(crypto.CipherVector, len(cv))

	for sourcePid := 1; sourcePid < netObj.GetNParty(); sourcePid++ {
		if netObj.GetPid() == sourcePid {
			cv = netObj.CollectiveBootstrapVec(cps, cv, sourcePid)
		} else {
			netObj.CollectiveBootstrapVec(cps, tmp, sourcePid)
		}
	}

	return cv
}

func (netObj ParallelNetworks) CollectiveRotKeyGen(parameters *ckks.Parameters, skShard *rlwe.SecretKey,
	crsVec []sampling.PRNG, rotTypes []crypto.RotationType) (rotKeys []*rlwe.GaloisKey) {

	slots := parameters.MaxSlots()

	shiftMap := make(map[int]bool)
	for _, rotType := range rotTypes {
		shift := rotType.Value
		if rotType.Side == crypto.SideRight {
			shift = slots - rotType.Value
		}
		shiftMap[shift] = true
	}

	gElems := make([]uint64, 0, len(shiftMap))
	for k := range shiftMap {
		gElems = append(gElems, parameters.GaloisElementForRotation(k))
	}
	gElems = append(gElems, parameters.GaloisElementForComplexConjugation())
	return netObj.CollectiveGaloisKeyGen(parameters, skShard, crsVec, gElems)
}

func (netObj ParallelNetworks) CollectiveGaloisKeyGen(parameters *ckks.Parameters, skShard *rlwe.SecretKey,
	crsVec []sampling.PRNG, requested []uint64) (rotKeys []*rlwe.GaloisKey) {

	set := make(map[uint64]bool, len(requested))
	for _, galEl := range requested {
		if galEl != 1 {
			set[galEl] = true
		}
	}
	gElems := make([]uint64, 0, len(set))
	for galEl := range set {
		gElems = append(gElems, galEl)
	}
	sort.Slice(gElems, func(i, j int) bool { return gElems[i] < gElems[j] })

	if strings.TrimSpace(os.Getenv("SFGWAS_SERIAL_ROTKEYGEN")) != "" {
		return netObj.collectiveRotKeyGenSerial(parameters, skShard, crsVec, gElems)
	}

	type rotJob struct {
		idx   int
		galEl uint64
	}

	rotKeys = make([]*rlwe.GaloisKey, len(gElems))
	nproc := len(netObj)
	jobChannels := make([]chan rotJob, nproc)
	for i := range jobChannels {
		jobChannels[i] = make(chan rotJob, 32)
	}

	go func() {
		for ind, galEl := range gElems {
			jobChannels[ind%nproc] <- rotJob{idx: ind, galEl: galEl}
			fmt.Println("Generate RotKey ", ind+1, "/", len(gElems), ", Galois element", galEl)
		}
		for _, c := range jobChannels {
			close(c)
		}
	}()

	var wg sync.WaitGroup
	for thread := 0; thread < nproc; thread++ {
		wg.Add(1)
		go func(thread int, net *Network, crs sampling.PRNG) {
			defer wg.Done()
			gkg := multiparty.NewGaloisKeyGenProtocol(*parameters)
			for job := range jobChannels[thread] {
				share := gkg.AllocateShare()
				crp := gkg.SampleCRP(crs)

				if net.GetPid() > 0 {
					if err := gkg.GenShare(skShard, job.galEl, crp, &share); err != nil {
						panic(err)
					}
				}

				agg := gkg.AllocateShare()
				agg.GaloisElement = job.galEl
				pid := net.GetPid()
				hubPid := net.GetHubPid()

				if pid == 0 {
					if hubPid > 0 {
						if err := agg.UnmarshalBinary(net.receiveBinaryValue(hubPid)); err != nil {
							panic(err)
						}
					}
				} else if pid == hubPid {
					for p := 1; p < net.GetNParty(); p++ {
						next := gkg.AllocateShare()
						if p == pid {
							next = share
						} else if err := next.UnmarshalBinary(net.receiveBinaryValue(p)); err != nil {
							panic(err)
						}
						if err := gkg.AggregateShares(next, agg, &agg); err != nil {
							panic(err)
						}
					}

					for p := 1; p < net.GetNParty(); p++ {
						if p != pid {
							net.sendBinaryValue(agg, p)
						}
					}
				} else {
					net.sendBinaryValue(share, hubPid)
					if err := agg.UnmarshalBinary(net.receiveBinaryValue(hubPid)); err != nil {
						panic(err)
					}
				}

				gk := rlwe.NewGaloisKey(*parameters)
				if err := gkg.GenGaloisKey(agg, crp, gk); err != nil {
					panic(err)
				}
				rotKeys[job.idx] = gk
			}
		}(thread, netObj[thread], crsVec[thread])
	}
	wg.Wait()

	return
}

func (netObj ParallelNetworks) collectiveRotKeyGenSerial(parameters *ckks.Parameters, skShard *rlwe.SecretKey,
	crsVec []sampling.PRNG, gElems []uint64) (rotKeys []*rlwe.GaloisKey) {

	rotKeys = make([]*rlwe.GaloisKey, len(gElems))
	net := netObj[0]
	crs := crsVec[0]
	gkg := multiparty.NewGaloisKeyGenProtocol(*parameters)

	for ind, galEl := range gElems {
		fmt.Println("Generate RotKey ", ind+1, "/", len(gElems), ", Galois element", galEl)

		share := gkg.AllocateShare()
		crp := gkg.SampleCRP(crs)
		if net.GetPid() > 0 {
			if err := gkg.GenShare(skShard, galEl, crp, &share); err != nil {
				panic(err)
			}
		}

		agg := gkg.AllocateShare()
		agg.GaloisElement = galEl
		pid := net.GetPid()
		hubPid := net.GetHubPid()

		if pid == 0 {
			if hubPid > 0 {
				if err := agg.UnmarshalBinary(net.receiveBinaryValue(hubPid)); err != nil {
					panic(err)
				}
			}
		} else if pid == hubPid {
			for p := 1; p < net.GetNParty(); p++ {
				next := gkg.AllocateShare()
				if p == pid {
					next = share
				} else if err := next.UnmarshalBinary(net.receiveBinaryValue(p)); err != nil {
					panic(err)
				}
				if err := gkg.AggregateShares(next, agg, &agg); err != nil {
					panic(err)
				}
			}

			for p := 1; p < net.GetNParty(); p++ {
				if p != pid {
					net.sendBinaryValue(agg, p)
				}
			}
		} else {
			net.sendBinaryValue(share, hubPid)
			if err := agg.UnmarshalBinary(net.receiveBinaryValue(hubPid)); err != nil {
				panic(err)
			}
		}

		gk := rlwe.NewGaloisKey(*parameters)
		if err := gkg.GenGaloisKey(agg, crp, gk); err != nil {
			panic(err)
		}
		rotKeys[ind] = gk
	}

	return rotKeys
}

func (netObj *Network) CollectiveRelinKeyGen(params *ckks.Parameters, skShard *rlwe.SecretKey, crs sampling.PRNG) (evk *rlwe.RelinearizationKey) {
	prot := multiparty.NewRelinearizationKeyGenProtocol(*params)
	ephSk, share1, share2 := prot.AllocateShare()
	crp := prot.SampleCRP(crs)

	if netObj.GetPid() > 0 {
		prot.GenShareRoundOne(skShard, crp, ephSk, &share1)
	}
	outRound1 := netObj.aggregateRelinearizationKeyGenShare(prot, share1)

	if netObj.GetPid() > 0 {
		prot.GenShareRoundTwo(ephSk, skShard, outRound1, &share2)
	}
	outRound2 := netObj.aggregateRelinearizationKeyGenShare(prot, share2)

	evk = rlwe.NewRelinearizationKey(*params)
	prot.GenRelinearizationKey(outRound1, outRound2, evk)
	return
}

func (netObj *Network) BroadcastCMat(cps *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid int, numCtxRow, numCtxCol int) crypto.CipherMatrix {
	return netObj.broadcastCMat(cps, cm, sourcePid, numCtxRow, numCtxCol, true)
}

// copySource preserves the public BroadcastCMat ownership contract. The
// rerandomizing path already owns a fresh matrix and can return it directly.
func (netObj *Network) broadcastCMat(cps *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid int, numCtxRow, numCtxCol int, copySource bool) crypto.CipherMatrix {
	if netObj.GetPid() == sourcePid {
		if len(cm) != numCtxRow || len(cm[0]) != numCtxCol {
			panic("BroadcastCVec: dimensions of cm do not match numCtxRow or numCtxCol")
		}

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != sourcePid {
				netObj.SendCipherMatrix(cm, p)
			}
		}
		if copySource {
			cm = crypto.CopyEncryptedMatrix(cm)
		}
	} else if netObj.GetPid() > 0 {
		cm = netObj.ReceiveCipherMatrix(cps, numCtxRow, numCtxCol, sourcePid)
	}
	return cm
}

// BroadcastCMatWithRandomZero rerandomizes at sourcePid before the first transfer.
// P1..PN, including the source, receive the same ciphertexts; P0 does not
// participate. Shape and source are public, and sourcePid must be a data site.
// Each data site must call this in the same public order. Inputs are preserved.
func (netObj *Network) BroadcastCMatWithRandomZero(cp *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid, rows, cols int) crypto.CipherMatrix {
	if sourcePid < 1 || sourcePid >= netObj.GetNParty() {
		panic("BroadcastCMatWithRandomZero: source must be a data site")
	}
	if netObj.GetPid() == sourcePid {
		cm = crypto.RerandomizeMatrix(cp, cm)
	}
	return netObj.broadcastCMat(cp, cm, sourcePid, rows, cols, false)
}

// TransferCMatWithRandomZero rerandomizes at sourcePid and sends only to targetPid.
// Only the target returns ciphertexts; all other parties return nil. Inputs are
// preserved. The endpoints and matrix dimensions are public data-site choices.
func (netObj *Network) TransferCMatWithRandomZero(cp *crypto.CryptoParams, cm crypto.CipherMatrix, sourcePid, targetPid, rows, cols int) crypto.CipherMatrix {
	if sourcePid < 1 || sourcePid >= netObj.GetNParty() || targetPid < 1 || targetPid >= netObj.GetNParty() || sourcePid == targetPid {
		panic("TransferCMatWithRandomZero: distinct data-site endpoints required")
	}
	if netObj.GetPid() == sourcePid {
		if len(cm) != rows {
			panic("TransferCMatWithRandomZero: row count mismatch")
		}
		for _, row := range cm {
			if len(row) != cols {
				panic("TransferCMatWithRandomZero: column count mismatch")
			}
		}
		netObj.SendCipherMatrix(crypto.RerandomizeMatrix(cp, cm), targetPid)
	} else if netObj.GetPid() == targetPid {
		return netObj.ReceiveCipherMatrix(cp, rows, cols, sourcePid)
	}
	return nil
}

func (netObj *Network) BroadcastCiphertext(cps *crypto.CryptoParams, ct *rlwe.Ciphertext, sourcePid int) *rlwe.Ciphertext {
	if netObj.GetPid() == sourcePid {
		for p := 1; p < netObj.GetNParty(); p++ {
			if p != sourcePid {
				netObj.SendCiphertext(ct, p)
			}
		}
		ct = ct.CopyNew()
	} else if netObj.GetPid() > 0 {
		ct = netObj.ReceiveCiphertext(cps, sourcePid)
	}
	return ct
}
