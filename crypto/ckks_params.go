package crypto

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/examples"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	CKKSParamsPN14QP427S38 = "PN14QP427S38"
	CKKSParamsPN14QP436S45 = "PN14QP436S45"
)

func IsPackedSKATParameters(name string) bool {
	return name == CKKSParamsPN14QP427S38 || name == CKKSParamsPN14QP436S45
}

// ResolveCKKSParametersLiteral is the shared source for supported CKKS profiles.
func ResolveCKKSParametersLiteral(name string) (ckks.ParametersLiteral, error) {
	switch name {
	case "PN12QP109":
		return examples.CKKSComplexParamsN12QP109, nil
	case "PN13QP218":
		return examples.CKKSComplexParamsN13QP218, nil
	case "PN14QP438":
		return examples.CKKSComplexParamsN14QP438, nil
	case CKKSParamsPN14QP427S38:
		return ckks.ParametersLiteral{
			LogN:            14,
			LogQ:            []int{40, 39, 38, 38, 38, 38, 38, 38},
			LogP:            []int{40, 40, 40},
			LogDefaultScale: 38,
		}, nil
	case CKKSParamsPN14QP436S45:
		return ckks.ParametersLiteral{
			LogN:            14,
			LogQ:            []int{56, 45, 45, 45, 45, 45, 45},
			LogP:            []int{55, 55},
			LogDefaultScale: 45,
		}, nil
	case "PN15QP880", "PN15QP881":
		return examples.CKKSComplexParamsN15QP881, nil
	case "PN16QP1761":
		return examples.CKKSComplexParamsPN16QP1761, nil
	default:
		return ckks.ParametersLiteral{}, fmt.Errorf("unsupported CKKS parameters %q", name)
	}
}
