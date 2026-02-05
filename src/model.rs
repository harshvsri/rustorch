use crate::tensor::Matrix;
use crate::{MNIST_IMG_SIZE, MNIST_LABEL_SIZE};
use bitflags::bitflags;
use std::io::{self, Write};

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

pub type ModelVarId = usize;

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

impl std::fmt::Display for ModelVarOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Null => "Null",
            Self::Create => "Create",
            Self::Relu => "Relu",
            Self::Softmax => "Softmax",
            Self::MatAdd => "MatAdd",
            Self::MatSub => "MatSub",
            Self::MatMul => "MatMul",
            Self::CrossEntropy => "CrossEntropy",
        };
        write!(f, "{name}")
    }
}

#[derive(Clone, Default)]
pub struct ModelVar {
    pub index: ModelVarId,
    pub flags: ModelVarFlags,
    pub op: ModelVarOperation,

    pub val: Option<Matrix>,
    pub grad: Option<Matrix>,
    /// This is the index for the nodes that were fed into this MedelVar
    pub inputs: [Option<ModelVarId>; Self::MAX_INPUTS],
}

impl ModelVar {
    pub const MAX_INPUTS: usize = 2;

    //  Create a fresh model variable and register it in the arena.
    pub fn create(
        model: &mut ModelContext,
        rows: usize,
        cols: usize,
        flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> ModelVarId {
        let mut out = Self::default();

        out.index = model.num_vars;
        model.num_vars += 1;
        out.flags = flags;
        out.op = op;
        out.val = Some(Matrix::new(rows, cols));

        if flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            out.grad = Some(Matrix::new(rows, cols));
        }

        //  HACK: Well here we can use the arena strategy and can use the index of the ModelVar rather
        // than cloning it over and over agian which will be quite inefficient.
        let index = out.index;
        model.vars.push(out);

        if flags.contains(ModelVarFlags::INPUT) {
            model.input = Some(index);
        }
        if flags.contains(ModelVarFlags::OUTPUT) {
            model.output = Some(index);
        }
        if flags.contains(ModelVarFlags::DESIRED_OUTPUT) {
            model.desired_output = Some(index);
        }
        if flags.contains(ModelVarFlags::COST) {
            model.cost = Some(index);
        }

        index
    }

    //  Unary operation plumbing with shape propagation.
    pub fn unary_implementation(
        model: &mut ModelContext,
        input: ModelVarId,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> ModelVarId {
        if model.vars[input]
            .flags
            .contains(ModelVarFlags::REQUIRES_GRAD)
        {
            flags.insert(ModelVarFlags::REQUIRES_GRAD);
        }

        let out = Self::create(model, rows, cols, flags, op);
        model.vars[out].inputs[0] = Some(input);
        out
    }

    //  Binary operation plumbing with shape propagation.
    pub fn binary_implementation(
        model: &mut ModelContext,
        a: ModelVarId,
        b: ModelVarId,
        rows: usize,
        cols: usize,
        mut flags: ModelVarFlags,
        op: ModelVarOperation,
    ) -> ModelVarId {
        if model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD)
            || model.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD)
        {
            flags.insert(ModelVarFlags::REQUIRES_GRAD);
        }

        let out = Self::create(model, rows, cols, flags, op);
        model.vars[out].inputs[0] = Some(a);
        model.vars[out].inputs[1] = Some(b);
        out
    }

    //  Register a ReLU node in the graph.
    pub fn relu(model: &mut ModelContext, input: ModelVarId, flags: ModelVarFlags) -> ModelVarId {
        let (rows, cols) = {
            let input_var = &model.vars[input];
            let val = input_var.val.as_ref().expect("Input must have value");
            (val.rows, val.cols)
        };

        Self::unary_implementation(model, input, rows, cols, flags, ModelVarOperation::Relu)
    }
    //  Register a Softmax node in the graph.
    pub fn softmax(
        model: &mut ModelContext,
        input: ModelVarId,
        flags: ModelVarFlags,
    ) -> ModelVarId {
        let (rows, cols) = {
            let input_var = &model.vars[input];
            let val = input_var.val.as_ref().expect("Input must have value");
            (val.rows, val.cols)
        };

        Self::unary_implementation(model, input, rows, cols, flags, ModelVarOperation::Softmax)
    }
    //  Register an Add node in the graph.
    pub fn add(
        model: &mut ModelContext,
        a: ModelVarId,
        b: ModelVarId,
        flags: ModelVarFlags,
    ) -> Option<ModelVarId> {
        let (rows, cols) = {
            let a_val = model.vars[a].val.as_ref().expect("Input must have value");
            let b_val = model.vars[b].val.as_ref().expect("Input must have value");
            if a_val.rows != b_val.rows || a_val.cols != b_val.cols {
                return None;
            }
            (a_val.rows, a_val.cols)
        };

        Some(Self::binary_implementation(
            model,
            a,
            b,
            rows,
            cols,
            flags,
            ModelVarOperation::MatAdd,
        ))
    }
    //  Register a Sub node in the graph.
    pub fn sub(
        model: &mut ModelContext,
        a: ModelVarId,
        b: ModelVarId,
        flags: ModelVarFlags,
    ) -> Option<ModelVarId> {
        let (rows, cols) = {
            let a_val = model.vars[a].val.as_ref().expect("Input must have value");
            let b_val = model.vars[b].val.as_ref().expect("Input must have value");
            if a_val.rows != b_val.rows || a_val.cols != b_val.cols {
                return None;
            }
            (a_val.rows, a_val.cols)
        };

        Some(Self::binary_implementation(
            model,
            a,
            b,
            rows,
            cols,
            flags,
            ModelVarOperation::MatSub,
        ))
    }
    //  Register a MatMul node in the graph.
    pub fn mul(
        model: &mut ModelContext,
        a: ModelVarId,
        b: ModelVarId,
        flags: ModelVarFlags,
    ) -> Option<ModelVarId> {
        let (rows, cols) = {
            let a_val = model.vars[a].val.as_ref().expect("Input must have value");
            let b_val = model.vars[b].val.as_ref().expect("Input must have value");
            if a_val.cols != b_val.rows {
                return None;
            }
            (a_val.rows, b_val.cols)
        };

        Some(Self::binary_implementation(
            model,
            a,
            b,
            rows,
            cols,
            flags,
            ModelVarOperation::MatMul,
        ))
    }
    //  Register a CrossEntropy node in the graph.
    pub fn cross_entropy(
        model: &mut ModelContext,
        a: ModelVarId,
        b: ModelVarId,
        flags: ModelVarFlags,
    ) -> Option<ModelVarId> {
        let (rows, cols) = {
            let a_val = model.vars[a].val.as_ref().expect("Input must have value");
            let b_val = model.vars[b].val.as_ref().expect("Input must have value");
            if a_val.rows != b_val.rows || a_val.cols != b_val.cols {
                return None;
            }
            (a_val.rows, a_val.cols)
        };

        Some(Self::binary_implementation(
            model,
            a,
            b,
            rows,
            cols,
            flags,
            ModelVarOperation::CrossEntropy,
        ))
    }
}

#[derive(Clone)]
pub struct ModelProgram {
    pub vars: Vec<ModelVarId>,
    pub size: usize,
}

impl Default for ModelProgram {
    fn default() -> Self {
        Self {
            vars: Vec::new(),
            size: 0,
        }
    }
}

pub struct ModelContext {
    pub num_vars: usize,
    pub vars: Vec<ModelVar>,
    pub forward_prog: ModelProgram,
    pub cost_program: ModelProgram,

    pub input: Option<ModelVarId>,
    pub output: Option<ModelVarId>,
    pub desired_output: Option<ModelVarId>,
    pub cost: Option<ModelVarId>,
}

impl Default for ModelContext {
    fn default() -> Self {
        Self {
            num_vars: 0,
            vars: Vec::new(),
            forward_prog: ModelProgram::default(),
            cost_program: ModelProgram::default(),
            input: None,
            output: None,
            desired_output: None,
            cost: None,
        }
    }
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

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn push(&mut self, node: ModelVar) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn get(&self, index: usize) -> Option<&ModelVar> {
        self.nodes.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut ModelVar> {
        self.nodes.get_mut(index)
    }
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

impl ModelTraininingDesc {
    pub fn new(
        train_images: Matrix,
        train_labels: Matrix,
        test_images: Matrix,
        test_labels: Matrix,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
    ) -> Self {
        Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            epochs,
            batch_size,
            learning_rate,
        }
    }
}

pub fn create_mnist_model(model: &mut ModelContext) {
    //  Build the MNIST feedforward graph (784-16-16-10) with residual add.
    //  Weight init uses a Xavier-style bound per layer width.
    //  The graph is purely declarative here; execution happens in the programs
    //  compiled from the output and cost nodes.
    let input = ModelVar::create(
        model,
        MNIST_IMG_SIZE,
        1,
        ModelVarFlags::INPUT,
        ModelVarOperation::Create,
    );

    let w0 = ModelVar::create(
        model,
        16,
        MNIST_IMG_SIZE,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );
    let w1 = ModelVar::create(
        model,
        16,
        16,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );
    let w2 = ModelVar::create(
        model,
        MNIST_LABEL_SIZE,
        16,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );

    let bound0 = (6.0f32 / (MNIST_IMG_SIZE as f32 + 16.0)).sqrt();
    let bound1 = (6.0f32 / (16.0 + 16.0)).sqrt();
    let bound2 = (6.0f32 / (16.0 + MNIST_LABEL_SIZE as f32)).sqrt();

    //  Initialize weights in a symmetric range to stabilize early training.
    if let Some(val) = model.vars[w0].val.as_mut() {
        val.fill_random_range(-bound0, bound0);
    }
    if let Some(val) = model.vars[w1].val.as_mut() {
        val.fill_random_range(-bound1, bound1);
    }
    if let Some(val) = model.vars[w2].val.as_mut() {
        val.fill_random_range(-bound2, bound2);
    }

    let b0 = ModelVar::create(
        model,
        16,
        1,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );
    let b1 = ModelVar::create(
        model,
        16,
        1,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );
    let b2 = ModelVar::create(
        model,
        MNIST_LABEL_SIZE,
        1,
        ModelVarFlags::REQUIRES_GRAD | ModelVarFlags::PARAMETER,
        ModelVarOperation::Create,
    );

    //  Layer 0: affine + ReLU
    let z0_a = ModelVar::mul(model, w0, input, ModelVarFlags::NONE).expect("W0*input failed");
    let z0_b = ModelVar::add(model, z0_a, b0, ModelVarFlags::NONE).expect("z0+b0 failed");
    let a0 = ModelVar::relu(model, z0_b, ModelVarFlags::NONE);

    //  Layer 1: affine + ReLU with a residual add from a0.
    let z1_a = ModelVar::mul(model, w1, a0, ModelVarFlags::NONE).expect("W1*a0 failed");
    let z1_b = ModelVar::add(model, z1_a, b1, ModelVarFlags::NONE).expect("z1+b1 failed");
    let z1_c = ModelVar::relu(model, z1_b, ModelVarFlags::NONE);
    let a1 = ModelVar::add(model, a0, z1_c, ModelVarFlags::NONE).expect("a0+z1 failed");

    //  Output layer: affine + softmax to produce class probabilities.
    let z2_a = ModelVar::mul(model, w2, a1, ModelVarFlags::NONE).expect("W2*a1 failed");
    let z2_b = ModelVar::add(model, z2_a, b2, ModelVarFlags::NONE).expect("z2+b2 failed");
    let output = ModelVar::softmax(model, z2_b, ModelVarFlags::OUTPUT);

    //  Desired output is a one-hot vector provided at training time.
    let y = ModelVar::create(
        model,
        MNIST_LABEL_SIZE,
        1,
        ModelVarFlags::DESIRED_OUTPUT,
        ModelVarOperation::Create,
    );

    //  Cost is computed against the desired labels.
    let _cost = ModelVar::cross_entropy(model, y, output, ModelVarFlags::COST)
        .expect("cross entropy failed");
}

pub fn model_prog_create(model: &ModelContext, out_var: ModelVarId) -> ModelProgram {
    //  Treat the variable arena as a DAG and compute a topological order.
    //  Returns a forward program starting at the output node.
    //  This mimics the C stack-based traversal to avoid recursion.
    let mut visited = vec![false; model.num_vars];
    let mut stack: Vec<ModelVarId> = Vec::with_capacity(model.num_vars);
    let mut out: Vec<ModelVarId> = Vec::with_capacity(model.num_vars);

    stack.push(out_var);

    while let Some(cur) = stack.pop() {
        if cur >= model.num_vars {
            continue;
        }

        if visited[cur] {
            if out.len() < model.num_vars {
                out.push(cur);
            }
            continue;
        }

        visited[cur] = true;
        if stack.len() < model.num_vars {
            stack.push(cur);
        }

        let num_inputs = model.vars[cur].op.num_inputs();
        for input_index in 0..num_inputs {
            let input = model.vars[cur].inputs[input_index];
            let Some(input) = input else { continue };

            if input >= model.num_vars || visited[input] {
                continue;
            }

            if let Some(existing) = stack.iter().position(|value| *value == input) {
                stack.remove(existing);
            }

            if stack.len() < model.num_vars {
                stack.push(input);
            }
        }
    }

    //  The resulting program is ordered for forward execution.
    ModelProgram {
        size: out.len(),
        vars: out,
    }
}

pub fn model_prog_compute(model: &mut ModelContext, prog: &ModelProgram) {
    //  Execute the forward program into each node's value buffer.
    //  Mirrors the C switch by dispatching on ModelVarOperation per node.
    //  Each op pulls its input values and writes into the current node's matrix.
    for i in 0..prog.size {
        let cur_index = prog.vars[i];
        let op = model.vars[cur_index].op.clone();
        let a_index = model.vars[cur_index].inputs[0];
        let b_index = model.vars[cur_index].inputs[1];

        match op {
            ModelVarOperation::Null | ModelVarOperation::Create => {}
            ModelVarOperation::Relu => {
                let a = a_index.expect("Relu expects input");
                let (input, out) = {
                    let input = model.vars[a].val.clone().expect("Input missing value");
                    let out = model.vars[cur_index]
                        .val
                        .as_mut()
                        .expect("Output missing value");
                    (input, out)
                };
                Matrix::relu_into(out, &input);
            }
            ModelVarOperation::Softmax => {
                let a = a_index.expect("Softmax expects input");
                let (input, out) = {
                    let input = model.vars[a].val.clone().expect("Input missing value");
                    let out = model.vars[cur_index]
                        .val
                        .as_mut()
                        .expect("Output missing value");
                    (input, out)
                };
                Matrix::softmax_into(out, &input);
            }
            ModelVarOperation::MatAdd => {
                let a = a_index.expect("Add expects input a");
                let b = b_index.expect("Add expects input b");
                let (a_val, b_val) = {
                    let a_val = model.vars[a].val.clone().expect("Input missing value");
                    let b_val = model.vars[b].val.clone().expect("Input missing value");
                    (a_val, b_val)
                };
                let out = model.vars[cur_index]
                    .val
                    .as_mut()
                    .expect("Output missing value");
                Matrix::add_into(out, &a_val, &b_val);
            }
            ModelVarOperation::MatSub => {
                let a = a_index.expect("Sub expects input a");
                let b = b_index.expect("Sub expects input b");
                let (a_val, b_val) = {
                    let a_val = model.vars[a].val.clone().expect("Input missing value");
                    let b_val = model.vars[b].val.clone().expect("Input missing value");
                    (a_val, b_val)
                };
                let out = model.vars[cur_index]
                    .val
                    .as_mut()
                    .expect("Output missing value");
                Matrix::sub_into(out, &a_val, &b_val);
            }
            ModelVarOperation::MatMul => {
                let a = a_index.expect("MatMul expects input a");
                let b = b_index.expect("MatMul expects input b");
                let (a_val, b_val) = {
                    let a_val = model.vars[a].val.clone().expect("Input missing value");
                    let b_val = model.vars[b].val.clone().expect("Input missing value");
                    (a_val, b_val)
                };
                let out = model.vars[cur_index]
                    .val
                    .as_mut()
                    .expect("Output missing value");
                Matrix::mul_into(out, &a_val, &b_val, true, false, false);
            }
            ModelVarOperation::CrossEntropy => {
                let a = a_index.expect("CrossEntropy expects input a");
                let b = b_index.expect("CrossEntropy expects input b");
                let (a_val, b_val) = {
                    let a_val = model.vars[a].val.clone().expect("Input missing value");
                    let b_val = model.vars[b].val.clone().expect("Input missing value");
                    (a_val, b_val)
                };
                let out = model.vars[cur_index]
                    .val
                    .as_mut()
                    .expect("Output missing value");
                Matrix::cross_entropy_into(out, &a_val, &b_val);
            }
        }
    }
}

pub fn model_prog_compute_grads(model: &mut ModelContext, prog: &ModelProgram) {
    //  Backpropagate gradients through the program in reverse order.
    //  Clears non-parameter grads, seeds the cost grad, then walks backwards.
    //  Parameter grads accumulate; non-parameters are reset each step.
    for i in 0..prog.size {
        let cur_index = prog.vars[i];
        let cur_flags = model.vars[cur_index].flags;

        if !cur_flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }

        if cur_flags.contains(ModelVarFlags::PARAMETER) {
            continue;
        }

        if let Some(grad) = model.vars[cur_index].grad.as_mut() {
            grad.clear();
        }
    }

    //  Seed the gradient of the final node (cost) to 1.
    if let Some(last_index) = prog.vars.last().copied() {
        if let Some(grad) = model.vars[last_index].grad.as_mut() {
            grad.fill(1.0);
        }
    }

    //  Reverse traversal applies the chain rule for each op.
    for idx in (0..prog.size).rev() {
        let cur_index = prog.vars[idx];
        let cur_flags = model.vars[cur_index].flags;
        if !cur_flags.contains(ModelVarFlags::REQUIRES_GRAD) {
            continue;
        }

        let op = model.vars[cur_index].op.clone();
        let a_index = model.vars[cur_index].inputs[0];
        let b_index = model.vars[cur_index].inputs[1];
        let num_inputs = op.num_inputs();

        if num_inputs == 1 {
            let a = a_index.expect("Unary op missing input");
            if !model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                continue;
            }
        }

        if num_inputs == 2 {
            let a = a_index.expect("Binary op missing input a");
            let b = b_index.expect("Binary op missing input b");
            if !model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD)
                && !model.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD)
            {
                continue;
            }
        }

        match op {
            ModelVarOperation::Null | ModelVarOperation::Create => {}
            ModelVarOperation::Relu => {
                let a = a_index.expect("Relu expects input");
                let (input, grad) = {
                    let input = model.vars[a].val.clone().expect("Input missing value");
                    let grad = model.vars[cur_index]
                        .grad
                        .clone()
                        .expect("Grad missing value");
                    (input, grad)
                };
                let a_grad = model.vars[a].grad.as_mut().expect("Input missing grad");
                Matrix::relu_add_grad(a_grad, &input, &grad);
            }
            ModelVarOperation::Softmax => {
                let a = a_index.expect("Softmax expects input");
                let (softmax_out, grad) = {
                    let softmax_out = model.vars[cur_index]
                        .val
                        .clone()
                        .expect("Output missing value");
                    let grad = model.vars[cur_index]
                        .grad
                        .clone()
                        .expect("Grad missing value");
                    (softmax_out, grad)
                };
                let a_grad = model.vars[a].grad.as_mut().expect("Input missing grad");
                Matrix::softmax_add_grad(a_grad, &softmax_out, &grad);
            }
            ModelVarOperation::MatAdd => {
                let a = a_index.expect("Add expects input a");
                let b = b_index.expect("Add expects input b");
                let grad = model.vars[cur_index]
                    .grad
                    .clone()
                    .expect("Grad missing value");
                if model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let a_grad = model.vars[a].grad.as_mut().expect("Input missing grad");
                    a_grad.add_assign(&grad);
                }
                if model.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let b_grad = model.vars[b].grad.as_mut().expect("Input missing grad");
                    b_grad.add_assign(&grad);
                }
            }
            ModelVarOperation::MatSub => {
                let a = a_index.expect("Sub expects input a");
                let b = b_index.expect("Sub expects input b");
                let grad = model.vars[cur_index]
                    .grad
                    .clone()
                    .expect("Grad missing value");
                if model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let a_grad = model.vars[a].grad.as_mut().expect("Input missing grad");
                    a_grad.add_assign(&grad);
                }
                if model.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let b_grad = model.vars[b].grad.as_mut().expect("Input missing grad");
                    b_grad.sub_assign(&grad);
                }
            }
            ModelVarOperation::MatMul => {
                let a = a_index.expect("MatMul expects input a");
                let b = b_index.expect("MatMul expects input b");
                let grad = model.vars[cur_index]
                    .grad
                    .clone()
                    .expect("Grad missing value");
                if model.vars[a].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let b_val = model.vars[b].val.clone().expect("Input missing value");
                    let a_grad = model.vars[a].grad.as_mut().expect("Input missing grad");
                    Matrix::mul_into(a_grad, &grad, &b_val, false, false, true);
                }
                if model.vars[b].flags.contains(ModelVarFlags::REQUIRES_GRAD) {
                    let a_val = model.vars[a].val.clone().expect("Input missing value");
                    let b_grad = model.vars[b].grad.as_mut().expect("Input missing grad");
                    Matrix::mul_into(b_grad, &a_val, &grad, false, true, false);
                }
            }
            ModelVarOperation::CrossEntropy => {
                let p = a_index.expect("CrossEntropy expects input p");
                let q = b_index.expect("CrossEntropy expects input q");
                let (p_val, q_val, grad) = {
                    let p_val = model.vars[p].val.clone().expect("Input missing value");
                    let q_val = model.vars[q].val.clone().expect("Input missing value");
                    let grad = model.vars[cur_index]
                        .grad
                        .clone()
                        .expect("Grad missing value");
                    (p_val, q_val, grad)
                };
                let p_grad = model.vars[p]
                    .grad
                    .as_mut()
                    .map(|value| value as *mut Matrix);
                let q_grad = model.vars[q]
                    .grad
                    .as_mut()
                    .map(|value| value as *mut Matrix);
                let p_grad = p_grad.map(|value| unsafe { &mut *value });
                let q_grad = q_grad.map(|value| unsafe { &mut *value });
                Matrix::cross_entropy_add_grad(p_grad, q_grad, &p_val, &q_val, &grad);
            }
        }
    }
}

pub fn model_create() -> ModelContext {
    //  Create a fresh model context with empty programs and no nodes.
    ModelContext::default()
}

pub fn model_compile(model: &mut ModelContext) {
    //  Compile forward and cost programs from output/cost roots.
    //  Mirrors the C `model_compile` behavior by building two programs.
    //  `forward_prog` runs inference; `cost_program` runs inference + loss.
    if let Some(output) = model.output {
        model.forward_prog = model_prog_create(model, output);
    }

    if let Some(cost) = model.cost {
        model.cost_program = model_prog_create(model, cost);
    }
}

pub fn model_feedforward(model: &mut ModelContext) {
    //  Run a forward pass using the compiled program.
    //  This updates model.output based on the current model.input values.
    let prog = model.forward_prog.clone();
    model_prog_compute(model, &prog);
}

pub fn model_train(model: &mut ModelContext, training_desc: &ModelTraininingDesc) {
    //  Train the model using mini-batch gradient descent.
    //  Shuffles each epoch, accumulates batch grads, then applies SGD.
    //  The training loop mirrors the C version: shuffle, batch, backprop, step.
    let train_images = &training_desc.train_images;
    let train_labels = &training_desc.train_labels;
    let test_images = &training_desc.test_images;
    let test_labels = &training_desc.test_labels;

    let num_examples = train_images.rows;
    let input_size = train_images.cols;
    let output_size = train_labels.cols;
    let num_tests = test_images.rows;

    let num_batches = num_examples / training_desc.batch_size;

    let mut training_order: Vec<usize> = (0..num_examples).collect();

    for epoch in 0..training_desc.epochs {
        for _ in 0..num_examples {
            let a = fastrand::usize(0..num_examples);
            let b = fastrand::usize(0..num_examples);
            training_order.swap(a, b);
        }

        for batch in 0..num_batches {
            for i in 0..model.cost_program.size {
                let cur_index = model.cost_program.vars[i];
                if model.vars[cur_index]
                    .flags
                    .contains(ModelVarFlags::PARAMETER)
                {
                    if let Some(grad) = model.vars[cur_index].grad.as_mut() {
                        grad.clear();
                    }
                }
            }

            let mut avg_cost = 0.0f32;
            for i in 0..training_desc.batch_size {
                let order_index = batch * training_desc.batch_size + i;
                let index = training_order[order_index];

                let input_index = model.input.expect("Model missing input");
                let output_index = model.desired_output.expect("Model missing desired output");

                if let Some(val) = model.vars[input_index].val.as_mut() {
                    let start = index * input_size;
                    let end = start + input_size;
                    val.data.copy_from_slice(&train_images.data[start..end]);
                }

                if let Some(val) = model.vars[output_index].val.as_mut() {
                    let start = index * output_size;
                    let end = start + output_size;
                    val.data.copy_from_slice(&train_labels.data[start..end]);
                }

                let prog = model.cost_program.clone();
                model_prog_compute(model, &prog);
                model_prog_compute_grads(model, &prog);

                let cost_index = model.cost.expect("Model missing cost");
                let cost_val = model.vars[cost_index]
                    .val
                    .as_ref()
                    .expect("Cost missing value");
                avg_cost += cost_val.sum();
            }
            avg_cost /= training_desc.batch_size as f32;

            for i in 0..model.cost_program.size {
                let cur_index = model.cost_program.vars[i];
                if !model.vars[cur_index]
                    .flags
                    .contains(ModelVarFlags::PARAMETER)
                {
                    continue;
                }

                let learning_rate = training_desc.learning_rate / training_desc.batch_size as f32;
                if let Some(grad) = model.vars[cur_index].grad.as_mut() {
                    grad.scale(learning_rate);
                }
                let grad = model.vars[cur_index]
                    .grad
                    .clone()
                    .expect("Grad missing value");
                if let Some(val) = model.vars[cur_index].val.as_mut() {
                    val.sub_assign(&grad);
                }
            }

            print!(
                "Epoch {:2} / {:2}, Batch {:4} / {:4}, Average Cost: {:.4}\r",
                epoch + 1,
                training_desc.epochs,
                batch + 1,
                num_batches,
                avg_cost
            );
            let _ = io::stdout().flush();
        }
        println!();

        let mut num_correct = 0usize;
        let mut avg_cost = 0.0f32;
        for i in 0..num_tests {
            let input_index = model.input.expect("Model missing input");
            let output_index = model.desired_output.expect("Model missing desired output");

            if let Some(val) = model.vars[input_index].val.as_mut() {
                let start = i * input_size;
                let end = start + input_size;
                val.data.copy_from_slice(&test_images.data[start..end]);
            }

            if let Some(val) = model.vars[output_index].val.as_mut() {
                let start = i * output_size;
                let end = start + output_size;
                val.data.copy_from_slice(&test_labels.data[start..end]);
            }

            let prog = model.cost_program.clone();
            model_prog_compute(model, &prog);

            let cost_index = model.cost.expect("Model missing cost");
            let output_var = model.output.expect("Model missing output");
            let cost_val = model.vars[cost_index]
                .val
                .as_ref()
                .expect("Cost missing value");
            let output_val = model.vars[output_var]
                .val
                .as_ref()
                .expect("Output missing value");
            let desired_val = model.vars[output_index]
                .val
                .as_ref()
                .expect("Desired missing value");
            avg_cost += cost_val.sum();
            if output_val.argmax() == desired_val.argmax() {
                num_correct += 1;
            }
        }

        avg_cost /= num_tests as f32;
        println!(
            "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
            num_correct,
            num_tests,
            num_correct as f32 / num_tests as f32 * 100.0,
            avg_cost
        );
    }
}
