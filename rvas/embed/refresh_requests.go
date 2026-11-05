package embed

import "github.com/hhcho/sfgwas/crypto"

const (
	refreshDone = iota
	refreshNext
)

// Adaptive hash circuits keep loop bounds at the source. Peers service one
// ciphertext per request until DONE, without receiving L, R, or batch lengths.
func (c *tableCircuit) sendRefreshCommand(command int) {
	for pid := 1; pid < c.net.GetNParty(); pid++ {
		if pid != c.source {
			c.net.SendInt(command, pid)
		}
	}
}

func (c *tableCircuit) serveRefreshRequests() {
	for {
		switch c.net.ReceiveInt(c.source) {
		case refreshDone:
			return
		case refreshNext:
			c.net.RefreshCMatWithRandomZero(c.cp, nil, c.source, 1, 1)
		default:
			panic("hash: unknown refresh command")
		}
	}
}

func (c *tableCircuit) requestRefresh(v crypto.CipherVector) crypto.CipherVector {
	out := make(crypto.CipherVector, len(v))
	for i, ct := range v {
		c.sendRefreshCommand(refreshNext)
		out[i] = c.net.RefreshCMatWithRandomZero(c.cp, crypto.CipherMatrix{{ct}}, c.source, 1, 1)[0][0]
	}
	return out
}
