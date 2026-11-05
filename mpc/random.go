package mpc

import (
	"fmt"
	"os"
	"path"

	"github.com/aead/chacha20/chacha"
	"github.com/hhcho/frand"
	mpc_core "github.com/hhcho/mpc-core"
)

type Random struct {
	pid      int
	prgTable map[int]*frand.RNG
	curPRG   *frand.RNG
}

const (
	bufferSize int = 1024
	GlobalPRG  int = -1
)

func sortInt(a, b int) (int, int) {
	if a < b {
		return a, b
	}
	return b, a
}

func readSharedPRGKey(sharedKeysPath, name string) ([]byte, error) {
	keyPath := path.Join(sharedKeysPath, name)
	key, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, fmt.Errorf("read shared PRG key %s: %w", keyPath, err)
	}
	if len(key) != chacha.KeySize {
		return nil, fmt.Errorf("shared PRG key %s has %d bytes, want exactly %d", keyPath, len(key), chacha.KeySize)
	}
	return key, nil
}

func mustReadSharedPRGKey(sharedKeysPath, name string) []byte {
	key, err := readSharedPRGKey(sharedKeysPath, name)
	if err != nil {
		panic(err)
	}
	return key
}

// InitializePRG loads shared keys; repeated initialization restarts their streams.
func InitializePRG(pid int, NumParties int, sharedKeysPath string) *Random {
	if sharedKeysPath == "" {
		panic("InitializePRG: shared_keys_path is required")
	}
	if pid < 0 || pid >= NumParties {
		panic("InitializePRG: invalid party ID")
	}

	prgTable := make(map[int]*frand.RNG, NumParties+1)
	seed := mustReadSharedPRGKey(sharedKeysPath, "shared_key_global.bin")
	prgTable[GlobalPRG] = frand.NewCustom(seed, bufferSize, 20)
	for other := 0; other < NumParties; other++ {
		if other == pid {
			continue
		}
		a, b := sortInt(pid, other)
		seed = mustReadSharedPRGKey(sharedKeysPath, fmt.Sprintf("shared_key_%d_%d.bin", a, b))
		prgTable[other] = frand.NewCustom(seed, bufferSize, 20)
	}

	// Local PRG remains freshly seeded on each initialization.
	seed = make([]byte, chacha.KeySize)
	frand.Read(seed)
	prgTable[pid] = frand.NewCustom(seed, bufferSize, 20)
	return &Random{
		prgTable: prgTable,
		curPRG:   prgTable[pid],
		pid:      pid,
	}
}

func (rand *Random) SwitchPRG(otherPid int) {
	rand.curPRG = rand.prgTable[otherPid]
}

func (rand *Random) RestorePRG() {
	rand.curPRG = rand.prgTable[rand.pid]
}

func (rand *Random) CurPRG() *frand.RNG {
	return rand.curPRG
}

func (rand *Random) ExportPRG() []byte {
	return nil
}

func (rand *Random) RandRead(buf []byte) {
	rand.curPRG.Read(buf)
}

func (rand *Random) RandElem(rtype mpc_core.RElem) mpc_core.RElem {
	return rtype.Rand(rand.curPRG)
}

func (rand *Random) RandVec(rtype mpc_core.RElem, n int) mpc_core.RVec {
	out := make(mpc_core.RVec, n)
	for i := range out {
		out[i] = rtype.Rand(rand.curPRG)
	}
	return out
}

func (rand *Random) RandMat(rtype mpc_core.RElem, r int, c int) mpc_core.RMat {
	out := make(mpc_core.RMat, r)
	for i := range out {
		out[i] = rand.RandVec(rtype, c)
	}
	return out
}

// Sample n random numbers with bit-length nbits in rtype
func (rand *Random) RandVecBits(rtype mpc_core.RElem, n, nbits int) mpc_core.RVec {
	out := make(mpc_core.RVec, n)
	for i := range out {
		out[i] = rtype.RandBits(rand.curPRG, nbits)
	}
	return out
}

// Sample random numbers with bit-length nbits in rtype
func (rand *Random) RandMatBits(rtype mpc_core.RElem, nrows, ncols, nbits int) mpc_core.RMat {
	out := make(mpc_core.RMat, nrows)
	for i := range out {
		out[i] = rand.RandVecBits(rtype, ncols, nbits)
	}
	return out
}
