package mat

// Tensor is the basic multi-dimentional tensor interface type.
type Tensor interface {
	// Dims returns the dimensions of a Tensor.
	Dims() []int

	// At returns the value of a matrix element at row i, column j.
	// It will panic if i or j are out of bounds for the matrix.
	At(i ...int) float64

	// T returns the transpose of the Tensor. Whether T returns a copy of the
	// underlying data is implementation dependent.
	T() Tensor
}
