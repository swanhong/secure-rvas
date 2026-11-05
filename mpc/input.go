package mpc

import (
	"crypto/rand"
	"encoding/binary"

	mpc_core "github.com/hhcho/mpc-core"
)

// ShareFromOwner inputs n values held by owner as additive shares across P1..PN.
// P0 receives zeros. All parties call with the same owner and public length n;
// only the owner needs to supply x. The input is not modified.
func (mpcObj *MPC) ShareFromOwner(owner int, x mpc_core.RVec, n int) mpc_core.RVec {
	if owner < 1 || owner >= mpcObj.GetNParty() {
		panic("ShareFromOwner: owner must be a data site")
	}
	if n < 0 {
		panic("ShareFromOwner: negative public length")
	}
	rtype, pid := mpcObj.GetRType(), mpcObj.GetPid()
	if pid == 0 {
		return mpc_core.InitRVec(rtype.Zero(), n)
	}
	stream := mpcObj.Network.Rand
	if pid != owner {
		stream.SwitchPRG(owner)
		pad := stream.RandVec(rtype, n)
		stream.RestorePRG()
		return pad
	}
	if len(x) != n {
		panic("ShareFromOwner: owner input length differs from public length")
	}
	share := x.Copy()
	for site := 1; site < mpcObj.GetNParty(); site++ {
		if site == owner {
			continue
		}
		stream.SwitchPRG(site)
		share.Sub(stream.RandVec(rtype, n))
		stream.RestorePRG()
	}
	return share
}

const hashKeyWords = 4

// DeriveHashKey returns a fresh 32-byte key common to the data sites, and nil at
// P0. P1 samples the key and masks each transfer with its pairwise PRG stream.
// Sites must call this and ShareFromOwner in the same public order.
func (mpcObj *MPC) DeriveHashKey() []byte {
	pid := mpcObj.GetPid()
	if pid == 0 {
		return nil
	}
	key := make([]byte, 8*hashKeyWords)
	if pid == 1 {
		if _, err := rand.Read(key); err != nil {
			panic(err)
		}
		for site := 2; site < mpcObj.GetNParty(); site++ {
			pad := mpcObj.hashKeyPad(site)
			masked := make([]uint64, hashKeyWords)
			for w := range masked {
				masked[w] = binary.BigEndian.Uint64(key[8*w:]) ^ binary.BigEndian.Uint64(pad[8*w:])
			}
			mpcObj.Network.SendIntVector(masked, site)
		}
		return key
	}
	masked := mpcObj.Network.ReceiveIntVector(hashKeyWords, 1)
	pad := mpcObj.hashKeyPad(1)
	for w, word := range masked {
		binary.BigEndian.PutUint64(key[8*w:], word^binary.BigEndian.Uint64(pad[8*w:]))
	}
	return key
}

func (mpcObj *MPC) hashKeyPad(other int) []byte {
	pad := make([]byte, 8*hashKeyWords)
	stream := mpcObj.Network.Rand
	stream.SwitchPRG(other)
	stream.RandRead(pad)
	stream.RestorePRG()
	return pad
}
