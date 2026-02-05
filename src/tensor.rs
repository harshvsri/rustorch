use crate::EPSILON;
use std::{error::Error, io::Read};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // Row major implementation
}

impl Matrix {
    /// Creates a new matrix with the given shape.
    ///
    /// The backing buffer is zero-filled so callers can write in-place without
    /// needing an explicit initialization step.
    pub fn new(rows: usize, cols: usize) -> Self {
        let size = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; size],
        }
    }

    /// Loads a matrix of `rows * cols` f32 values from a binary file.
    ///
    /// If the file is smaller than the expected size, the remaining entries
    /// stay at zero.
    pub fn load(rows: usize, cols: usize, filename: &'static str) -> Result<Self, Box<dyn Error>> {
        let mut file = std::fs::File::open(filename)?;
        let meta = file.metadata()?;
        let max_size = std::cmp::min(rows * cols, (meta.len() / 4) as usize);

        let mut buf = vec![0u8; max_size * 4]; // f32 -> 4 Bytes
        file.read_exact(&mut buf)?;

        let mut data = vec![0.0; rows * cols];
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("Chunk size must be of exactly length 4.");
            data[i] = f32::from_le_bytes(bytes);
        }

        Ok(Matrix { rows, cols, data })
    }

    /// Returns the total number of elements in the matrix.
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Fills the matrix with random values in [0, f32::MAX).
    ///
    /// This is primarily useful for quick smoke tests and debugging.
    pub fn fill_random(&mut self) {
        if self.data.len() != self.size() {
            self.data.resize(self.size(), 0.0);
        }
        for value in &mut self.data {
            *value = fastrand::f32() * f32::MAX;
        }
    }

    /// Fills the matrix with random values in [lower, upper).
    ///
    /// Useful for symmetric weight initialization.
    pub fn fill_random_range(&mut self, lower: f32, upper: f32) {
        if self.data.len() != self.size() {
            self.data.resize(self.size(), 0.0);
        }
        for value in &mut self.data {
            *value = fastrand::f32() * (upper - lower) + lower;
        }
    }

    /// Sets all elements to zero without changing the shape.
    pub fn clear(&mut self) {
        for value in &mut self.data {
            *value = 0.0;
        }
    }

    /// Fills the matrix with a constant value.
    pub fn fill(&mut self, value: f32) {
        for item in &mut self.data {
            *item = value;
        }
    }

    /// Returns the sum of all elements.
    ///
    /// This is used for computing average cost values in training.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Copies values from `src` into `self` when shapes match.
    ///
    /// Returns `false` if the dimensions differ.
    pub fn copy(&mut self, src: &Self) -> bool {
        if self.rows != src.rows || self.cols != src.cols {
            return false;
        }

        self.data.copy_from_slice(&src.data);
        true
    }

    /// Scales all elements by the given factor.
    ///
    /// Commonly used to apply the learning rate to gradients.
    pub fn scale(&mut self, factor: f32) {
        for i in 0..self.size() {
            self.data[i] *= factor;
        }
    }

    /// Adds another matrix into this one in-place.
    ///
    /// Useful for gradient accumulation.
    pub fn add_assign(&mut self, other: &Self) {
        for i in 0..self.size() {
            self.data[i] += other.data[i];
        }
    }

    /// Subtracts another matrix from this one in-place.
    ///
    /// Useful for SGD parameter updates.
    pub fn sub_assign(&mut self, other: &Self) {
        for i in 0..self.size() {
            self.data[i] -= other.data[i];
        }
    }

    /// Returns the index of the maximum element.
    ///
    /// Used for classification accuracy checks.
    pub fn argmax(&self) -> usize {
        let mut max_i = 0;
        for i in 1..self.size() {
            if self.data[i] > self.data[max_i] {
                max_i = i;
            }
        }
        max_i
    }

    /// Transposes the matrix in-place by materializing a new buffer.
    ///
    /// This swaps rows and columns and rewrites the underlying storage.
    pub fn transpose(&mut self) {
        let (new_rows, new_cols) = (self.cols, self.rows);
        let mut new_data = Vec::with_capacity(self.size());
        for c in 0..self.cols {
            for r in 0..self.rows {
                new_data.push(self.data[r * self.cols + c]);
            }
        }

        self.rows = new_rows;
        self.cols = new_cols;
        self.data = new_data;
    }

    /// Multiplies this matrix by `other` and returns a new matrix.
    ///
    /// Returns `None` when the inner dimensions do not match.
    pub fn matmul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }

        let mut mat = Matrix::new(self.rows, other.cols);
        Matrix::mul_into(&mut mat, self, other, true, false, false);
        Some(mat)
    }

    /// Returns a new matrix containing the element-wise sum.
    ///
    /// Returns `None` when shapes differ.
    pub fn matsum(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        Matrix::add_into(&mut mat, self, other);
        Some(mat)
    }

    /// Returns a new matrix containing the element-wise difference.
    ///
    /// Returns `None` when shapes differ.
    pub fn matsub(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        Matrix::sub_into(&mut mat, self, other);
        Some(mat)
    }

    /// Applies the ReLU (Rectified Linear Unit) activation function to the matrix.
    ///
    /// ReLU is a non-linear activation function defined as f(x) = max(0, x).
    /// It effectively sets all negative values to zero while keeping positive values unchanged.
    pub fn relu(&self) -> Self {
        let mut mat = self.clone();
        Matrix::relu_into(&mut mat, self);
        mat
    }

    /// Applies the (Naive) Softmax function to the matrix.
    ///
    /// Softmax converts a vector of raw scores (logits) into a probability distribution.
    /// The elements of the resulting matrix will be in the range (0, 1) and sum up to 1.0.
    pub fn softmax(&self) -> Self {
        let mut mat = self.clone();
        Matrix::softmax_into(&mut mat, self);
        mat
    }

    /// Calculates the Cross Entropy Loss (element-wise) between this matrix (targets) and another (predictions).
    ///
    /// This metric measures the difference between two probability distributions.
    /// In classification, 'self' is usually the "One Hot Encoded" true label,
    /// and 'other' is the probability output from the Softmax function.
    pub fn cross_entropy(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Matrix::new(self.rows, self.cols);
        Matrix::cross_entropy_into(&mut mat, self, other);
        Some(mat)
    }

    /// Writes the element-wise sum into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn add_into(out: &mut Self, a: &Self, b: &Self) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        if out.rows != a.rows || out.cols != a.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] = a.data[i] + b.data[i];
        }

        true
    }

    /// Writes the element-wise difference into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn sub_into(out: &mut Self, a: &Self, b: &Self) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        if out.rows != a.rows || out.cols != a.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] = a.data[i] - b.data[i];
        }

        true
    }

    /// Multiplies A * B without transposing either operand.
    fn mat_mul_nn(out: &mut Self, a: &Self, b: &Self) {
        for i in 0..out.rows {
            for k in 0..a.cols {
                let a_val = a.data[k + i * a.cols];
                for j in 0..out.cols {
                    out.data[j + i * out.cols] += a_val * b.data[j + k * b.cols];
                }
            }
        }
    }

    /// Multiplies A * B^T (right operand transposed).
    fn mat_mul_nt(out: &mut Self, a: &Self, b: &Self) {
        for i in 0..out.rows {
            for j in 0..out.cols {
                let mut sum = 0.0;
                for k in 0..a.cols {
                    sum += a.data[k + i * a.cols] * b.data[k + j * b.cols];
                }
                out.data[j + i * out.cols] += sum;
            }
        }
    }

    /// Multiplies A^T * B (left operand transposed).
    fn mat_mul_tn(out: &mut Self, a: &Self, b: &Self) {
        for k in 0..a.rows {
            for i in 0..out.rows {
                let a_val = a.data[i + k * a.cols];
                for j in 0..out.cols {
                    out.data[j + i * out.cols] += a_val * b.data[j + k * b.cols];
                }
            }
        }
    }

    /// Multiplies A^T * B^T (both operands transposed).
    fn mat_mul_tt(out: &mut Self, a: &Self, b: &Self) {
        for i in 0..out.rows {
            for j in 0..out.cols {
                let mut sum = 0.0;
                for k in 0..a.rows {
                    sum += a.data[i + k * a.cols] * b.data[k + j * b.cols];
                }
                out.data[j + i * out.cols] += sum;
            }
        }
    }

    /// Multiplies matrices into `out` with optional transposes.
    ///
    /// The transpose flags select one of four variants:
    /// nn: A * B, nt: A * B^T, tn: A^T * B, tt: A^T * B^T.
    pub fn mul_into(
        out: &mut Self,
        a: &Self,
        b: &Self,
        zero_out: bool,
        transpose_a: bool,
        transpose_b: bool,
    ) -> bool {
        let a_rows = if transpose_a { a.cols } else { a.rows };
        let a_cols = if transpose_a { a.rows } else { a.cols };
        let b_rows = if transpose_b { b.cols } else { b.rows };
        let b_cols = if transpose_b { b.rows } else { b.cols };

        if a_cols != b_rows {
            return false;
        }
        if out.rows != a_rows || out.cols != b_cols {
            return false;
        }

        if zero_out {
            out.clear();
        }

        let transpose = ((transpose_a as u8) << 1) | (transpose_b as u8);
        match transpose {
            0b00 => Matrix::mat_mul_nn(out, a, b),
            0b01 => Matrix::mat_mul_nt(out, a, b),
            0b10 => Matrix::mat_mul_tn(out, a, b),
            0b11 => Matrix::mat_mul_tt(out, a, b),
            _ => {}
        }

        true
    }

    /// Applies ReLU into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn relu_into(out: &mut Self, input: &Self) -> bool {
        if out.rows != input.rows || out.cols != input.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] = 0.0f32.max(input.data[i]);
        }

        true
    }

    /// Applies naive softmax into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn softmax_into(out: &mut Self, input: &Self) -> bool {
        if out.rows != input.rows || out.cols != input.cols {
            return false;
        }

        let mut sum = 0.0f32;
        for i in 0..out.size() {
            out.data[i] = input.data[i].exp();
            sum += out.data[i];
        }

        if sum != 0.0 {
            out.scale(1.0 / sum);
        }

        true
    }

    /// Computes element-wise cross entropy into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn cross_entropy_into(out: &mut Self, p: &Self, q: &Self) -> bool {
        if p.rows != q.rows || p.cols != q.cols {
            return false;
        }
        if out.rows != p.rows || out.cols != p.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] = if p.data[i] == 0.0 {
                0.0
            } else {
                p.data[i] * -(q.data[i] + EPSILON).ln()
            };
        }

        true
    }

    /// Adds the ReLU gradient into `out`.
    ///
    /// Returns `false` when shapes do not match.
    pub fn relu_add_grad(out: &mut Self, input: &Self, grad: &Self) -> bool {
        if out.rows != input.rows || out.cols != input.cols {
            return false;
        }
        if out.rows != grad.rows || out.cols != grad.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] += if input.data[i] > 0.0 {
                grad.data[i]
            } else {
                0.0
            };
        }

        true
    }

    /// Applies the softmax Jacobian to the upstream gradient.
    ///
    /// Returns `false` when the input is not a vector.
    pub fn softmax_add_grad(out: &mut Self, softmax_out: &Self, grad: &Self) -> bool {
        if softmax_out.rows != 1 && softmax_out.cols != 1 {
            return false;
        }

        let size = softmax_out.rows.max(softmax_out.cols);
        let mut jacobian = Matrix::new(size, size);
        for i in 0..size {
            for j in 0..size {
                jacobian.data[j + i * size] =
                    softmax_out.data[i] * ((i == j) as i32 as f32 - softmax_out.data[j]);
            }
        }

        Matrix::mul_into(out, &jacobian, grad, true, false, false)
    }

    /// Adds cross-entropy gradients into `p_grad` and `q_grad` if provided.
    ///
    /// Returns `false` when shapes do not match.
    pub fn cross_entropy_add_grad(
        p_grad: Option<&mut Self>,
        q_grad: Option<&mut Self>,
        p: &Self,
        q: &Self,
        grad: &Self,
    ) -> bool {
        if p.rows != q.rows || p.cols != q.cols {
            return false;
        }

        let size = p.size();

        if let Some(p_grad) = p_grad {
            if p_grad.rows != p.rows || p_grad.cols != p.cols {
                return false;
            }
            for i in 0..size {
                p_grad.data[i] += -(q.data[i] + EPSILON).ln() * grad.data[i];
            }
        }

        if let Some(q_grad) = q_grad {
            if q_grad.rows != q.rows || q_grad.cols != q.cols {
                return false;
            }
            for i in 0..size {
                q_grad.data[i] += -p.data[i] / (q.data[i] + EPSILON) * grad.data[i];
            }
        }

        true
    }
}
