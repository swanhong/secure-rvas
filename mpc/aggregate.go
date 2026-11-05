package mpc

import (
	"github.com/hhcho/sfgwas/crypto"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func (netObj *Network) AggregateDecryptShareVec(cryptoParams *crypto.CryptoParams, share []ring.Poly, level int) []ring.Poly {
	out := make([]ring.Poly, len(share))
	pid := netObj.GetPid()
	if pid == 0 {
		return out
	}

	ringQ := cryptoParams.Params.RingQ().AtLevel(level)
	hubPid := netObj.GetHubPid()

	if pid == hubPid {
		for i := range share {
			out[i] = *share[i].CopyNew()
		}
		for p := 1; p < netObj.GetNParty(); p++ {
			if p == pid {
				continue
			}
			for i := range share {
				other := netObj.ReceivePoly(p)
				ringQ.Add(out[i], other, out[i])
			}
		}
		for p := 1; p < netObj.GetNParty(); p++ {
			if p == pid {
				continue
			}
			for i := range out {
				netObj.SendPoly(out[i], p)
			}
		}
	} else {
		for i := range share {
			netObj.SendPoly(share[i], hubPid)
		}
		for i := range out {
			out[i] = netObj.ReceivePoly(hubPid)
		}
	}

	return out
}

func (netObj *Network) AggregateCText(cryptoParams *crypto.CryptoParams, val *rlwe.Ciphertext) (out *rlwe.Ciphertext) {
	pid := netObj.GetPid()
	if pid == 0 {
		return nil
	}

	if pid == netObj.GetHubPid() {
		// receive and add
		out = val.CopyNew()
		if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
			for p := 1; p < netObj.GetNParty(); p++ {
				if p != pid {
					other := netObj.ReceiveCiphertext(cryptoParams, p)
					if err := eval.Add(other, out, out); err != nil {
						return err
					}
				}
			}
			return nil
		}); err != nil {
			panic(err)
		}

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.SendCiphertext(out, p)
			}
		}
	} else {
		netObj.SendCiphertext(val, netObj.GetHubPid())
		out = netObj.ReceiveCiphertext(cryptoParams, netObj.GetHubPid())
	}

	return
}

func (netObj *Network) AggregateIntVec(vec []uint64) (out []uint64) {
	pid := netObj.GetPid()
	if pid == 0 {
		return nil
	}

	if pid == netObj.GetHubPid() {
		// receive and add
		out = make([]uint64, len(vec))
		copy(out, vec)

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != pid {
				other := netObj.ReceiveIntVector(len(vec), p)
				for i := range other {
					out[i] += other[i]
				}
			}
		}

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.SendIntVector(out, p)
			}
		}
	} else {
		netObj.SendIntVector(vec, netObj.GetHubPid())
		out = netObj.ReceiveIntVector(len(vec), netObj.GetHubPid())
	}

	return
}

func (netObj *Network) AggregateSharesCT(cryptoParams *crypto.CryptoParams, ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	if netObj.GetPid() == 0 {
		return nil
	}
	agg := ct.CopyNew()

	// Get shares from everyone except for pid = 0.
	pid := netObj.GetPid()
	for i := 1; i < netObj.GetNParty(); i++ {
		if pid == i {
			continue
		} else if i > pid {
			// receive first and then send
			agg = crypto.Add(cryptoParams, agg, netObj.ReceiveCiphertext(cryptoParams, i))
			netObj.SendCiphertext(ct, i)
		} else {
			// send and then receive
			netObj.SendCiphertext(ct, i)
			agg = crypto.Add(cryptoParams, agg, netObj.ReceiveCiphertext(cryptoParams, i))
		}
	}
	return agg
}

func (netObj *Network) AggregateCVec(cryptoParams *crypto.CryptoParams, vec crypto.CipherVector) crypto.CipherVector {
	return netObj.AggregateCMat(cryptoParams, crypto.CipherMatrix{vec})[0]
}

func (netObj *Network) AggregateCMat(cryptoParams *crypto.CryptoParams, mat crypto.CipherMatrix) (out crypto.CipherMatrix) {
	pid := netObj.GetPid()
	if pid == 0 {
		return nil
	}

	if pid == netObj.GetHubPid() {
		// receive and add
		out = crypto.CopyEncryptedMatrix(mat)
		if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
			for p := 1; p < netObj.GetNParty(); p++ {
				if p != pid {
					other := netObj.ReceiveCipherMatrix(cryptoParams, len(out), len(out[0]), p)
					for i := range other {
						for j := range other[i] {
							if err := eval.Add(other[i][j], out[i][j], out[i][j]); err != nil {
								return err
							}
						}
					}
				}
			}
			return nil
		}); err != nil {
			panic(err)
		}

		for p := 1; p < netObj.GetNParty(); p++ {
			if p != pid {
				netObj.SendCipherMatrix(out, p)
			}
		}
	} else {
		netObj.SendCipherMatrix(mat, netObj.GetHubPid())
		out = netObj.ReceiveCipherMatrix(cryptoParams, len(mat), len(mat[0]), netObj.GetHubPid())
	}

	return
}
