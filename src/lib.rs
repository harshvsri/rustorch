pub mod tensor;

use crate::tensor::Matrix;
use bitflags::bitflags;

pub const EPSILON: f32 = 1e-15;
pub const MNIST_IMG_DIMENTION: usize = 28;
pub const MNIST_IMG_SIZE: usize = MNIST_IMG_DIMENTION * MNIST_IMG_DIMENTION;
pub const MNIST_LABEL_SIZE: usize = 10;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct ModelVarFlags: u8 {
        const NONE           = 0;
        const REQUIRES_GRAD  = 1 << 0;
        const PARAMETER      = 1 << 1;
        const INPUT          = 1 << 2;
        const OUTPUT         = 1 << 3;
        const DESIRED_OUTPUT = 1 << 4;
        const COST           = 1 << 5;
    }
}

#[derive(Clone, Default)]
pub enum ModelVarOperation {
    #[default]
    Null,
    Create,
    Relu,
    Softmax,
    MatAdd,
    MatSub,
    MatMul,
    CrossEntropy,
}

impl ModelVarOperation {
    pub fn num_inputs(&self) -> usize {
        match self {
            Self::Null | Self::Create => 0,
            Self::Relu | Self::Softmax => 1,
            Self::MatAdd | Self::MatSub | Self::MatMul | Self::CrossEntropy => 2,
        }
    }
}

#[derive(Clone, Default)]
pub struct ModelVar {
    pub index: usize,
    pub flags: ModelVarFlags,
    pub op: ModelVarOperation,

    pub val: Option<Matrix>,
    pub grad: Option<Matrix>,
    /// This is the index for the nodes that were fed into this MedelVar
    pub inputs: [Option<usize>; Self::MAX_INPUTS],
}

//  FIX: Remember to remove this temporary attribute
#[allow(unused_variables)]
impl ModelVar {
    pub const MAX_INPUTS: usize = 2;

    pub fn create(
        model: &mut ModelContext,
        rows: usize,
        cols: usize,
        flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> Self {
        let mut out = Self::default();

        out.index = model.num_vars;
        model.num_vars += 1;
        out.flags = flags;
        out.op = ModelVarOperation::Create;
        out.val = Some(Matrix::new(rows, cols));

        if flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            out.grad = Some(Matrix::new(rows, cols));
        }

        //  HACK: Well here we can use the arena strategy and can use the index of the ModelVar rather
        // than cloning it over and over agian which will be quite inefficient.
        if flags.contains(ModelVarFlags::INPUT) {
            model.input = out.clone();
        }
        if flags.contains(ModelVarFlags::OUTPUT) {
            model.output = out.clone();
        }
        if flags.contains(ModelVarFlags::DESIRED_OUTPUT) {
            model.desired_output = out.clone();
        }
        if flags.contains(ModelVarFlags::COST) {
            model.cost = out.clone();
        }
        out
    }

    pub fn unary_implementation(
        model: &mut ModelContext,
        input: Self,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> Self {
        if input.flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            flags.insert(ModelVarFlags::REQUIRES_GRAD);
        }

        let mut out = Self::default();
        out.op = op;
        //  BUG: This expects to take index of the ModelVar which will be a part of the arena.
        out.inputs[0] = Some(0);
        out
    }

    pub fn binary_implementation(
        model: &mut ModelContext,
        a: Self,
        b: Self,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> Self {
        if a.flags.contains(ModelVarFlags::REQUIRES_GRAD)
            || b.flags.contains(ModelVarFlags::REQUIRES_GRAD)
        {
            flags.insert(ModelVarFlags::REQUIRES_GRAD);
        }

        let mut out = Self::default();

        out.op = op;
        //  BUG: This expects to take index of the ModelVar which will be a part of the arena.
        out.inputs[0] = Some(0);
        out.inputs[1] = Some(0);
        out
    }

    pub fn relu(model: &ModelContext, input: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
    pub fn softmax(model: &ModelContext, input: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
    pub fn add(model: &ModelContext, a: Self, b: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
    pub fn sub(model: &ModelContext, a: Self, b: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
    pub fn mul(model: &ModelContext, a: Self, b: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
    pub fn cross_entropy(model: &ModelContext, a: Self, b: Self, flags: ModelVarFlags) -> Self {
        Self::default()
    }
}

pub struct ModelProgram {
    pub vars: Vec<ModelVar>,
    pub size: usize,
}

pub struct ModelContext {
    pub num_vars: usize,
    pub forward_prog: ModelProgram,
    pub cost_program: ModelProgram,

    pub input: ModelVar,
    pub output: ModelVar,
    pub desired_output: ModelVar,
    pub cost: ModelVar,
}

/// A computational graph managed using the **Arena (or Pool) Pattern**.
///
/// In idiomatic Rust, linked lists and graphs are difficult to implement with
/// raw pointers or `Box` due to strict ownership rules. This structure solves
/// that by flattening the graph into a single vector.
///
/// # How it works
/// * **Ownership:** The `Graph` struct holds sole ownership of all nodes via the `nodes` vector.
/// * **Linking:** Instead of pointers (e.g., `&ModelVar`), nodes refer to each other using
///   integer indices (`usize`) into the `nodes` vector.
///
/// # Benefits
/// * **Borrow Checker Friendly:** Bypasses complex lifetime (`'a`) tracking and `Rc/RefCell` overhead.
/// * **Cache Locality:** Since `Vec` stores items contiguously in memory, traversing the graph
///   is significantly faster due to fewer CPU cache misses.
pub struct Graph {
    /// New nodes are pushed to the end of this vector, and their index
    /// (the result of `nodes.len() - 1`) is returned as their "handle" or "ID".
    pub nodes: Vec<ModelVar>,
}

pub struct ModelTraininingDesc {
    pub train_images: Matrix,
    pub train_labels: Matrix,
    pub test_images: Matrix,
    pub test_labels: Matrix,

    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
}
