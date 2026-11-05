package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
)

// FactorSPD returns the Cholesky factor of a shared SPD matrix and the inverses of its diagonal.
func FactorSPD(
	mpcObj *mpc.MPC,
	matrix mpc_core.RMat,
) (
	lower mpc_core.RMat,
	diagInv mpc_core.RVec,
) {
	size, _ := matrix.Dims()
	rtype := matrix.Type()

	lower = mpc_core.InitRMat(rtype.Zero(), size, size)
	diagInv = mpc_core.InitRVec(rtype.Zero(), size)

	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()
	useBoolean := mpcObj.GetBooleanShareFlag()

	for column := 0; column < size; column++ {
		pivot := matrix[column][column].Copy()
		if column > 0 {
			squares := mpcObj.SSSquareElemVec(lower[column][:column])
			squares = mpcObj.TruncVec(squares, dataBits, fracBits)

			for _, square := range squares {
				pivot = pivot.Sub(square)
			}
		}

		pivotSqrt, pivotSqrtInv := mpcObj.SqrtAndSqrtInverse(
			mpc_core.RVec{pivot},
			useBoolean,
		)
		lower[column][column] = pivotSqrt[0]
		diagInv[column] = pivotSqrtInv[0]

		rowsBelow := size - column - 1
		if rowsBelow == 0 {
			continue
		}

		numerators := mpc_core.InitRVec(rtype.Zero(), rowsBelow)
		for offset := range numerators {
			row := column + 1 + offset
			numerators[offset] = matrix[row][column].Copy()
		}

		if column > 0 {
			left := make(mpc_core.RVec, 0, rowsBelow*column)
			right := make(mpc_core.RVec, 0, rowsBelow*column)

			for row := column + 1; row < size; row++ {
				left = append(left, lower[row][:column]...)
				right = append(right, lower[column][:column]...)
			}

			products := mpcObj.SSMultElemVec(left, right)
			products = mpcObj.TruncVec(products, dataBits, fracBits)

			for offset := range numerators {
				start := offset * column
				for _, product := range products[start : start+column] {
					numerators[offset] = numerators[offset].Sub(product)
				}
			}
		}

		columnValues := mpcObj.SSMultElemVecScalar(
			numerators,
			diagInv[column],
		)
		columnValues = mpcObj.TruncVec(
			columnValues,
			dataBits,
			fracBits,
		)

		for offset, value := range columnValues {
			lower[column+1+offset][column] = value
		}
	}

	return lower, diagInv
}

func SolveSPD(
	mpcObj *mpc.MPC,
	lower mpc_core.RMat,
	diagInv mpc_core.RVec,
	rhs mpc_core.RMat,
) mpc_core.RMat {
	size, _ := lower.Dims()
	_, rhsColumns := rhs.Dims()
	rtype := rhs.Type()

	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()

	forward := mpc_core.InitRMat(rtype.Zero(), size, rhsColumns)
	for row := 0; row < size; row++ {
		values := rhs[row].Copy()

		if row > 0 {
			left := make(mpc_core.RVec, 0, row*rhsColumns)
			right := make(mpc_core.RVec, 0, row*rhsColumns)

			for inner := 0; inner < row; inner++ {
				for column := 0; column < rhsColumns; column++ {
					left = append(left, lower[row][inner])
					right = append(right, forward[inner][column])
				}
			}

			products := mpcObj.SSMultElemVec(left, right)
			products = mpcObj.TruncVec(products, dataBits, fracBits)

			for inner := 0; inner < row; inner++ {
				start := inner * rhsColumns
				for column := 0; column < rhsColumns; column++ {
					values[column] = values[column].Sub(
						products[start+column],
					)
				}
			}
		}

		values = mpcObj.SSMultElemVecScalar(values, diagInv[row])
		forward[row] = mpcObj.TruncVec(values, dataBits, fracBits)
	}

	solved := mpc_core.InitRMat(rtype.Zero(), size, rhsColumns)
	for row := size - 1; row >= 0; row-- {
		values := forward[row].Copy()
		rowsBelow := size - row - 1

		if rowsBelow > 0 {
			left := make(mpc_core.RVec, 0, rowsBelow*rhsColumns)
			right := make(mpc_core.RVec, 0, rowsBelow*rhsColumns)

			for inner := row + 1; inner < size; inner++ {
				for column := 0; column < rhsColumns; column++ {
					left = append(left, lower[inner][row])
					right = append(right, solved[inner][column])
				}
			}

			products := mpcObj.SSMultElemVec(left, right)
			products = mpcObj.TruncVec(products, dataBits, fracBits)

			for offset := 0; offset < rowsBelow; offset++ {
				start := offset * rhsColumns
				for column := 0; column < rhsColumns; column++ {
					values[column] = values[column].Sub(
						products[start+column],
					)
				}
			}
		}

		values = mpcObj.SSMultElemVecScalar(values, diagInv[row])
		solved[row] = mpcObj.TruncVec(values, dataBits, fracBits)
	}

	return solved
}
