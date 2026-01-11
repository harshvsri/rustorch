use std::{error::Error, io::Read};

pub const EPSILON: f32 = 1e-15;
pub const MNIST_IMG_DIMENTION: usize = 28;
pub const MNIST_IMG_SIZE: usize = MNIST_IMG_DIMENTION * MNIST_IMG_DIMENTION;
pub const MNIST_LABEL_SIZE: usize = 10;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // Row major implementation
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: Vec::with_capacity(rows * cols),
        }
    }

    pub fn load(rows: usize, cols: usize, filename: &'static str) -> Result<Self, Box<dyn Error>> {
        let mut file = std::fs::File::open(filename)?;
        let meta = file.metadata()?;
        let max_size = std::cmp::min(rows * cols, meta.len() as usize);

        let mut buf = vec![0u8; max_size * 4]; // f32 -> 4 Bytes
        file.read_exact(&mut buf)?;

        let data = buf
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk
                    .try_into()
                    .expect("Chunk size must be of exactly lenght 4.");
                f32::from_le_bytes(bytes) // Interpret as float
            })
            .collect::<Vec<_>>();

        Ok(Matrix { rows, cols, data })
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn fill_random(&mut self) {
        for _ in 0..(self.size()) {
            self.data.push(fastrand::f32() * f32::MAX); // Range(0..1 -> 0..f32::MAX)
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn copy(&mut self, src: &Self) -> bool {
        if self.rows != src.rows || self.cols != src.cols {
            return false;
        }

        self.data.copy_from_slice(&src.data);
        return true;
    }

    pub fn scale(&mut self, factor: f32) {
        for i in 0..self.size() {
            self.data[i] *= factor;
        }
    }

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

    pub fn matmul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }

        let mut mat = Matrix::new(self.rows, other.cols);
        for i in 0..mat.rows {
            for j in 0..mat.cols {
                let mut sum = 0f32;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                mat.data.push(sum);
            }
        }
        Some(mat)
    }

    pub fn matsum(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        for i in 0..self.size() {
            mat.data.push(self.data[i] + other.data[i]);
        }
        Some(mat)
    }

    pub fn matsub(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        for i in 0..self.size() {
            mat.data.push(self.data[i] - other.data[i]);
        }
        Some(mat)
    }

    /// Applies the ReLU (Rectified Linear Unit) activation function to the matrix.
    ///
    /// ReLU is a non-linear activation function defined as f(x) = max(0, x).
    /// It effectively sets all negative values to zero while keeping positive values unchanged.
    pub fn relu(&self) -> Self {
        let mut mat = self.clone();
        for i in 0..mat.size() {
            mat.data[i] = 0f32.max(mat.data[i]);
        }
        mat
    }

    /// Applies the (Naive) Softmax function to the matrix.
    ///
    /// Softmax converts a vector of raw scores (logits) into a probability distribution.
    /// The elements of the resulting matrix will be in the range (0, 1) and sum up to 1.0.
    pub fn softmax(&self) -> Self {
        let mut mat = self.clone();
        let mut sum = 0f32;
        for i in 0..mat.size() {
            mat.data[i] = mat.data[i].exp();
            sum += mat.data[i];
        }
        mat.scale(1f32 / sum);
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
        for i in 0..self.size() {
            mat.data.push(if self.data[i] == 0f32 {
                0f32
            } else {
                self.data[i] * -(other.data[i] + EPSILON).ln() // Adding a tiny value to prevent log(0)
            })
        }
        Some(mat)
    }
}
