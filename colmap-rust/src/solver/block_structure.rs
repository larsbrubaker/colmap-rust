// The block structure of a block-sparse Jacobian: runs of rows or columns (blocks), the
// stored cells of each row block, and the whole compressed-row layout. Port of the structure
// types in colmap-sharp's `ColmapSharp/Solver/BlockSparseMatrix.cs` (our MIT code), which
// follow Ceres Solver 2.2.0 `internal/ceres/block_structure.h` (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// One row block per residual block, one column block per (variable) parameter block, and a
// dense row-major cell wherever a residual block depends on a parameter block. These are plain
// data with public fields, as Ceres' structs are: the evaluator builds them from the program,
// the Schur eliminator walks them chunk by chunk, and the block random access matrices
// (`block_random_access_matrix.rs`) reuse [`Block`] for the F blocks of the reduced camera
// matrix. `block_sparse_matrix.rs` owns the values the cells point into.

/// `ceres::internal::Block`: a run of `size` rows or columns starting at `position`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block {
    /// Number of rows or columns.
    pub size: usize,
    /// Index of the first row or column.
    pub position: usize,
}

impl Block {
    /// A block of `size` starting at `position`.
    pub fn new(size: usize, position: usize) -> Self {
        Self { size, position }
    }
}

/// `ceres::internal::Cell`: a stored block in a row block, by column block and by the offset
/// of its first value in the matrix's value array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    /// The column block.
    pub block_id: usize,
    /// Offset of the cell's (0, 0) value; the cell is row-major, row block size x column
    /// block size.
    pub position: usize,
}

impl Cell {
    /// The cell of column block `block_id` whose values start at `position`.
    pub fn new(block_id: usize, position: usize) -> Self {
        Self { block_id, position }
    }
}

/// `ceres::internal::CompressedRow`: a row block and its cells, sorted by column block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedRow {
    /// The rows.
    pub block: Block,
    /// The stored cells, by increasing column block.
    pub cells: Vec<Cell>,
}

impl CompressedRow {
    /// A row block with its cells.
    pub fn new(block: Block, cells: Vec<Cell>) -> Self {
        Self { block, cells }
    }
}

/// `ceres::internal::CompressedRowBlockStructure`: the column blocks and the row blocks with
/// their cells.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompressedRowBlockStructure {
    /// The column blocks, by increasing position.
    pub cols: Vec<Block>,
    /// The row blocks, by increasing position.
    pub rows: Vec<CompressedRow>,
}

impl CompressedRowBlockStructure {
    /// The structure with these column and row blocks.
    pub fn new(cols: Vec<Block>, rows: Vec<CompressedRow>) -> Self {
        Self { cols, rows }
    }
}
