package embed

import "fmt"

// Setting is the public table shape (M, rho, sigma, H).
type Setting struct {
	M, Rho, Sigma, H int
}

func (p Setting) validate() error {
	r := p.Rho + p.Sigma
	if p.M <= 0 || p.M&(p.M-1) != 0 || p.Rho < 0 || p.Sigma < 0 || r < p.M || r&(r-1) != 0 {
		return fmt.Errorf("table requires power-of-two M and R, M <= R, rho >= 0, sigma >= 0")
	}
	if (p.Rho == 0 && (p.Sigma != p.M || p.H != 0)) || (p.Rho > 0 && p.H <= 0) {
		return fmt.Errorf("expected dense (0,M,0), or hash with rho,H > 0")
	}
	return nil
}
