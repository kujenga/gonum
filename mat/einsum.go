package mat

import (
	"bufio"
	"fmt"
	"io"
	"strings"
	"unicode"
)

// Einsum executes operations defined using einstein summation notation on the
// passed in operands, allowing for compactly defined operations to be
// performed on matrices.
func Einsum(
	subscripts string,
	operands ...Matrix,
) ([]int, []float64) {
	ops := parseEinsum(subscripts)

	exc, dim := ops.executor(operands)

	out := ops.outputZeros(dim)
	// Iterate over the counter states in the executor until it reaches
	// completion, incrementing values within the output at each step.
	for ; !exc.done; exc.increment() {
		// Index of the output value we are currently modifying.
		outputIdx := exc.outputIdx()
		// Compute the current value within the "inner loop" we are
		// constructing here with our counters. For each operand, we
		// multiply the output by the appropriate value. Starting point
		// is one because the value is arrived at multiplicatively.
		var cur float64 = 1
		for i, o := range operands {
			// Get current counter values for the runes,
			// representing the location in the output that
			// we want to modify.
			indexes := make([]int, 0, 2)
			for _, r := range ops.inputs[i] {
				indexes = append(indexes,
					exc.counterValFor(r))
			}

			// NOTE: This only allows 1D Vector or 2D Matrix inputs
			// at present, as that is what the Matrix data
			// structure supports in gonum. This could be extended
			// quite easily to work with a function signature like
			// .At(i ...int) for an N-dimensional array.
			if len(indexes) == 1 {
				cur *= o.At(indexes[0], 0)
			} else {
				cur *= o.At(indexes[0], indexes[1])
			}
		}
		// Add the resulting multiplied value to the summation.
		out[outputIdx] += cur
	}

	return dim, out
}

// Captures the operations requested within a parsed einsum instruction,
// facilitating their execution.
type einsumOps struct {
	// sequence of all the characters within the input sequence in raw
	// form. Used for various "look behind" behaviors in parsing, generally
	// not needed after parsing is completed.
	seq []rune
	// Holds the set of all the runes used in the einsum subscripts.
	all map[rune]bool
	// Holds all "free" indices within the einsum subscripts, which are the
	// ones we will iterate over when computing the output. Non-free
	// indices may be called "summation" indices, as values along thm are
	// summed.
	free map[rune]bool

	// Array of arrays of the comma-separated inputs, e.g.
	// "ij,jk" goes to: [][]rune{{'i', 'j'}, {'j', 'k'}}
	inputs [][]rune
	// Array of the indices specified in the outputs, e.g.
	// "ik" goes to: []rune{'i', 'k'}
	output []rune
}

// einsumExecutor is used to control operation and execution of the einsum
// operation, holding counters for the matrix iteration execution which emulate
// the appropriate number of loops for a hand-coded summation, and keeping
// track of overall progress and state.
type einsumExecutor struct {
	// Holds counters for the execution of the summation operation. These
	// can be thought of like an old-school car odometer, where each
	// counter in the array will rotate through it's max value and then
	// flip back to zero. By construction, once the counter is "maxed out"
	// fully the summation operation is completed.
	//
	// The reason for having this construct is to emulate an arbitrarily
	// nested depth of for loops, which is needed to be able to support
	// arbitrary dimentions of arrays, as the numpy implementation does.
	c []einsumCounter

	// Set to true when the execution is complete, indicating that the
	// counters have all been maxed out.
	done bool
}

func (c *einsumExecutor) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "{done: %v, ", c.done)
	for i, ec := range c.c {
		b.WriteString(ec.String())
		if i < len(c.c)-1 {
			b.WriteString(", ")
		}
	}
	b.WriteRune('}')
	return b.String()
}

// outputIdx returns the index in the output for the current state of the
// executor counters. This provides the 1D projection of the ND output array.
func (c *einsumExecutor) outputIdx() int {
	var outputIdx int
	jumpSize := 1
	for i := range c.c {
		// iterate backwards
		ec := c.c[len(c.c)-(i+1)]
		// Skip over the non-free indexes
		if !ec.free {
			continue
		}
		// Increment the index to the correct location, assuming rows
		// are flattened end-on-end.
		outputIdx += ec.v * jumpSize
		jumpSize *= ec.dim
	}
	return outputIdx
}

// counterFor returns the counter for the given rune.
func (c *einsumExecutor) counterValFor(r rune) int {
	// NOTE: Using a map here does not make things faster, because of the
	// limited number of values.
	for _, c := range c.c {
		if c.r == r {
			return c.v
		}
	}
	panic(fmt.Errorf("einsumCounter not found for rune: %q", r))
}

// einsumCounter provides a counter for the operations
type einsumCounter struct {
	// Contains the rune character representing the index for this counter.
	r rune
	// Indicates whether this is a "free" index that is multiplied, or a
	// summation index.
	free bool
	v    int
	dim  int
}

func (ec einsumCounter) String() string {
	return fmt.Sprintf("{rune: %q, free: %t, value: %d, dim: %d}",
		ec.r, ec.free, ec.v, ec.dim)
}

// executor initializes the executor for the parsed ops and operands, which
// includes an array of counters, representing the indices of the loop for
// computation of the einsum.
//
// It returns the executor itself, the dimensions of the output array.
func (o einsumOps) executor(operands []Matrix) (*einsumExecutor, []int) {
	c := &einsumExecutor{
		c: make([]einsumCounter, len(o.all)),
	}
	// First add free indices for multiplication. This forms the "outer
	// layers" of the emulated nested loop.
	for i, r := range o.output {
		c.c[i] = einsumCounter{
			r:    r,
			free: true,
			v:    0,
		}
	}
	// Then add all summation indexes to the end of the counter, representing
	// the "inner layers" the emulated nested loop. Order does not really
	// matter here as the summation is commutative, so we iterate over the
	// map of all index runes directly.
	s := 0
	for r := range o.all {
		if o.free[r] {
			continue
		}
		c.c[len(o.output)+s] = einsumCounter{
			r:    r,
			free: false,
			v:    0,
		}
		s++
	}

	// After they are constructed, set the dimension values for all
	// outputs, and construct an array of output dimensions for free
	// indices.
	var outputDim []int
	for i := range c.c {
		dim := o.dimOf(c.c[i].r, operands)
		c.c[i].dim = dim
		if c.c[i].free {
			outputDim = append(outputDim, dim)
		}
	}
	return c, outputDim
}

// Increments the counters for the next step in the execution loop.
func (c *einsumExecutor) increment() {
	// Iterate backwards through the array, incrementing the lowest counter
	// or otherwise resetting maxed ones.
	for i := range c.c {
		idx := len(c.c) - (i + 1)
		if c.c[idx].v+1 < c.c[idx].dim {
			// Normally, the lowest counter is incremented and we
			// continue.
			c.c[idx].v++
			return
		} else {
			// If this counter has reached the max, either return
			// if it is the highest-level counter, or reset it if
			// it is an inner one.
			if idx == 0 {
				c.done = true
				return
			}
			c.c[idx].v = 0
			// Continue looping to increment the counter above.
		}
	}
}

// dimOf returns the dimension of the output for the indicated rune index. It
// does this by looking at the input specification to determine where the rune
// was mentioned, and then mapping that to the dimension in the corresponding
// operand.
func (o einsumOps) dimOf(r rune, operands []Matrix) int {
	var dim int
	for i := range o.inputs {
		match := -1
		for l := range o.inputs[i] {
			if o.inputs[i][l] == r {
				match = l
				break
			}
		}
		var v int
		// Set the dimension to the observed length
		switch match {
		case -1:
			// Skip inputs where it is not matched.
			continue
		case 0:
			// Row length
			v, _ = operands[i].Dims()
		case 1:
			// Col length
			_, v = operands[i].Dims()
		default:
			// This is another location where an NDArray that
			// returns Dims() as []int would be able to support
			// higher dimensionality.
			panic(fmt.Errorf("only 2D matrix supported, %d indicated", match))
		}
		if dim == 0 {
			// If the dimension is unset, we set it.
			dim = v
		} else {
			// If the dimension has already been set, we
			// make sure it is the same as the currently
			// set value.
			if dim != v {
				panic(fmt.Errorf("expected dimension %d did not match %d for input %d", dim, v, i))

			}
		}
	}
	return dim
}

// outputZeros returns an initialized array of zero-valued float64 elements of
// the size needed to return the 1D projection of the ND output array.
func (o einsumOps) outputZeros(dim []int) []float64 {
	total := 1
	for _, d := range dim {
		total *= d
	}
	return make([]float64, total)
}

func (o einsumOps) String() string {
	var b strings.Builder
	for x := range o.inputs {
		for y := range o.inputs[x] {
			b.WriteRune(o.inputs[x][y])
		}
		if x < len(o.inputs)-1 {
			b.WriteRune(',')
		}
	}
	b.WriteString("->")
	for _, o := range o.output {
		b.WriteRune(o)
	}
	return b.String()
}

// StringWithExecutor renders a string representation of the einsum with
// included counters, useful for debugging purposes.
func (o einsumOps) StringWithExecutor(exc *einsumExecutor) string {
	var b strings.Builder
	w := func(r rune) {
		b.WriteRune(r)
		v := exc.counterValFor(r)
		fmt.Fprintf(&b, "(%d)", v)
	}
	for x := range o.inputs {
		for y := range o.inputs[x] {
			w(o.inputs[x][y])
		}
		if x < len(o.inputs)-1 {
			b.WriteRune(',')
		}
	}
	b.WriteString("->")
	for _, o := range o.output {
		w(o)
	}
	return b.String()
}

// Basic state tracker used in iteration through the einsum subscripts.
type opsParseMode int

const (
	// A new input is being started. This is the starting state.
	opsParseModeNewInput opsParseMode = iota
	// Growing an existing input.
	opsParseModeGrowInput
	// Adding to the output.
	opsParseModeOutput
)

// parseEinsum performs parsing of the einsum subscripts, producing a
// data structure form that is used for executing the computation.
func parseEinsum(subscripts string) einsumOps {
	o := einsumOps{
		all:  make(map[rune]bool),
		free: make(map[rune]bool),
	}

	var mode opsParseMode
	rdr := bufio.NewReader(strings.NewReader(subscripts))
	for {
		r, _, err := rdr.ReadRune()
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(fmt.Errorf("unexpected error reading string: %v", err))
		}

		switch r {
		case ',':
			// Indicates a new input
			mode = opsParseModeNewInput
		case '-':
			// store in seq and wait for >
		case '>':
			if o.last() != '-' {
				panic(fmt.Errorf("unexpected char %q after '-'", r))
			}
			// Moving to output phase.
			mode = opsParseModeOutput
		default:
			if unicode.IsSpace(r) {
				// Space characters are ignored.
				continue
			}
			if !unicode.IsLetter(r) {
				panic(fmt.Errorf("unexpected non-letter character: %q", r))
			}
			// Record the rune in the list of all runes.
			o.all[r] = true
			// Slot the rune into the correct location.
			switch mode {
			case opsParseModeNewInput:
				o.inputs = append(o.inputs, []rune{r})
				mode = opsParseModeGrowInput
			case opsParseModeGrowInput:
				o.inputs[len(o.inputs)-1] = append(
					o.inputs[len(o.inputs)-1], r)
			case opsParseModeOutput:
				o.free[r] = true
				o.output = append(o.output, r)
			}
		}

		o.seq = append(o.seq, r)
	}
	return o
}

// last returns the last value in the sequence of runes defining the einsum
// operations. Useful primarily in parsing.
func (o einsumOps) last() rune {
	if len(o.seq) == 0 {
		return 0
	}
	return o.seq[len(o.seq)-1]
}
