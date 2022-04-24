package mat

import (
	"testing"

	"golang.org/x/exp/rand"
)

func TestEinsum(t *testing.T) {
	// Various structures that can be used in tests.
	a := NewVecDense(2, []float64{
		1, 2,
	})
	b := NewVecDense(2, []float64{
		3, 4,
	})
	A := NewDense(2, 2, []float64{
		1, 2,
		3, 4,
	})
	B := NewDense(2, 2, []float64{
		1, 0,
		1, 1,
	})
	C := NewDense(2, 2, []float64{
		2, 3,
		4, 5,
	})

	for i, tc := range []struct {
		subscripts string
		operands   []Matrix
		expectDim  []int
		expectOut  []float64
	}{
		{
			// Element-wise Multiplication
			subscripts: "i,i->i",
			operands:   []Matrix{a, b},
			expectDim:  []int{2},
			expectOut: []float64{
				3, 8,
			},
		},
		{
			// Outer product
			subscripts: "i,j->ij",
			operands:   []Matrix{a, b},
			expectDim:  []int{2, 2},
			expectOut: []float64{
				3, 4,
				6, 8,
			},
		},
		{
			// Diagonal
			subscripts: "ii->i",
			operands: []Matrix{
				A,
			},
			expectDim: []int{2},
			expectOut: []float64{
				1, 4,
			},
		},
		{
			// Element-wise Multiplication
			subscripts: "ij,ij->ij",
			operands: []Matrix{
				B,
				C,
			},
			expectDim: []int{2, 2},
			expectOut: []float64{
				2, 0,
				4, 5,
			},
		},
		{
			// Element-wise Multiplication with Transpose
			subscripts: "ij,ji->ij",
			operands: []Matrix{
				B,
				C,
			},
			expectDim: []int{2, 2},
			expectOut: []float64{
				2, 0,
				3, 5,
			},
		},
		{
			// Element-wise Multiplication for three arrays
			subscripts: "ij,ij,ij->ij",
			operands: []Matrix{
				A,
				B,
				C,
			},
			expectDim: []int{2, 2},
			expectOut: []float64{
				2, 0,
				12, 20,
			},
		},
		{
			// Matrix Multiplication
			subscripts: "ik,kj->ij",
			operands: []Matrix{
				B,
				C,
			},
			expectDim: []int{2, 2},
			expectOut: []float64{
				2, 3,
				6, 8,
			},
		},
		{
			// Matrix Multiplication
			subscripts: "ik,kj->ij",
			operands: []Matrix{
				B,
				C,
			},
			expectDim: []int{2, 2},
			expectOut: []float64{
				2, 3,
				6, 8,
			},
		},
		{
			// Rectangular Matrix Multiplication
			subscripts: "ik,kj->ij",
			operands: []Matrix{
				NewDense(4, 3, []float64{
					1, 2, 5,
					3, 4, 3,
					7, 2, 9,
					1, 4, 2,
				}),
				NewDense(3, 4, []float64{
					1, 0, 2, 1,
					1, 1, 0, 5,
					8, 3, 1, 7,
				}),
			},
			expectDim: []int{4, 4},
			expectOut: []float64{
				43, 17, 7, 46,
				31, 13, 9, 44,
				81, 29, 23, 80,
				21, 10, 4, 35,
			},
		},
		{
			// Bigger Matrix Multiplication
			subscripts: "ik,kj->ij",
			operands: []Matrix{
				NewDense(4, 4, []float64{
					1, 2, 5, 2,
					3, 4, 3, 5,
					7, 2, 9, 4,
					1, 4, 2, 0,
				}),
				NewDense(4, 4, []float64{
					1, 0, 2, 1,
					1, 1, 0, 5,
					8, 3, 1, 7,
					0, 1, 5, 2,
				}),
			},
			expectDim: []int{4, 4},
			expectOut: []float64{
				43, 19, 17, 50,
				31, 18, 34, 54,
				81, 33, 43, 88,
				21, 10, 4, 35,
			},
		},
		{
			// each row of A multiplied by B
			subscripts: "ij,kj->ikj",
			operands: []Matrix{
				B,
				C,
			},
			expectDim: []int{2, 2, 2},
			expectOut: []float64{
				2, 0,
				4, 0,
				2, 3,
				4, 5,
			},
		},
	} {
		dim, got := Einsum(tc.subscripts, tc.operands...)
		if got == nil {
			t.Errorf("unexpected nil of Einsum for test %d", i)
			continue
		}
		var expectM, gotM Matrix
		switch len(dim) {
		case 1:
			expectM = NewVecDense(tc.expectDim[0], tc.expectOut)
			gotM = NewVecDense(dim[0], got)
		case 2:
			expectM = NewDense(tc.expectDim[0], tc.expectDim[1], tc.expectOut)
			gotM = NewDense(dim[0], dim[1], got)
		default:
			// For higher dimensions, we just compare as raw vectors.
			expectM = NewVecDense(len(tc.expectOut), tc.expectOut)
			gotM = NewVecDense(len(got), got)
			continue
		}
		if !Equal(gotM, expectM) {
			t.Errorf("matrix does not equal expected for test %d: got: %v want: %v", i, gotM, expectM)
		}
	}
}

func TestEinsumParse(t *testing.T) {
	for i, tc := range []struct {
		subscripts string
	}{
		{
			subscripts: "i,i->i",
		},
		{
			subscripts: "ij,jk->ik",
		},
		{
			subscripts: "ij,jk,kl->ikl",
		},
	} {
		ops := parseEinsum(tc.subscripts)
		str := ops.String()
		if tc.subscripts != str {
			t.Errorf("different string for parsed Einsum for test %d: got: %v expect: %v", i, str, tc.subscripts)
			continue
		}
	}
}

func BenchmarkEinsumMatMul10(b *testing.B)   { einsumBench(b, "ij,jk->ik", 10) }
func BenchmarkEinsumMatMul100(b *testing.B)  { einsumBench(b, "ij,jk->ik", 100) }
func BenchmarkEinsumMatMul1000(b *testing.B) { einsumBench(b, "ij,jk->ik", 1000) }

func einsumBench(b *testing.B, subscripts string, size int) {
	src := rand.NewSource(1)
	A, _ := randDense(size, 1, src)
	B, _ := randDense(size, 1, src)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Einsum(subscripts, A, B)
	}
}
