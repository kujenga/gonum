package mat_test

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExampleEinsum() {
	A := mat.NewDense(2, 2, []float64{
		1, 2,
		3, 4,
	})
	B := mat.NewDense(2, 2, []float64{
		1, 0,
		1, 1,
	})

	// Matrix multiplication
	dim, output := mat.Einsum("ij,jk->ik", A, B)
	C := mat.NewDense(dim[0], dim[1], output)
	fmt.Printf("Einsum result:\n%1.1f\n\n", mat.Formatted(C))
	// Output:
	// Einsum result:
	// ⎡3.0  2.0⎤
	// ⎣7.0  4.0⎦
}
