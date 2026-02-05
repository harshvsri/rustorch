use crate::EPSILON;
use std::{error::Error, io::Read};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // Row major implementation
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let size = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; size],
        }
    }

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

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn fill_random(&mut self) {
        if self.data.len() != self.size() {
            self.data.resize(self.size(), 0.0);
        }
        for value in &mut self.data {
            *value = fastrand::f32() * f32::MAX;
        }
    }

    pub fn fill_random_range(&mut self, lower: f32, upper: f32) {
        if self.data.len() != self.size() {
            self.data.resize(self.size(), 0.0);
        }
        for value in &mut self.data {
            *value = fastrand::f32() * (upper - lower) + lower;
        }
    }

    pub fn clear(&mut self) {
        for value in &mut self.data {
            *value = 0.0;
        }
    }

    pub fn fill(&mut self, value: f32) {
        for item in &mut self.data {
            *item = value;
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn copy(&mut self, src: &Self) -> bool {
        if self.rows != src.rows || self.cols != src.cols {
            return false;
        }

        self.data.copy_from_slice(&src.data);
        true
    }

    pub fn scale(&mut self, factor: f32) {
        for i in 0..self.size() {
            self.data[i] *= factor;
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        for i in 0..self.size() {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub_assign(&mut self, other: &Self) {
        for i in 0..self.size() {
            self.data[i] -= other.data[i];
        }
    }

    pub fn argmax(&self) -> usize {
        let mut max_i = 0;
        for i in 1..self.size() {
            if self.data[i] > self.data[max_i] {
                max_i = i;
            }
        }
        max_i
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
        Matrix::mul_into(&mut mat, self, other, true, false, false);
        Some(mat)
    }

    pub fn matsum(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        Matrix::add_into(&mut mat, self, other);
        Some(mat)
    }

    pub fn matsub(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Self::new(self.rows, self.cols);
        Matrix::sub_into(&mut mat, self, other);
        Some(mat)
    }

    pub fn relu(&self) -> Self {
        let mut mat = self.clone();
        Matrix::relu_into(&mut mat, self);
        mat
    }

    pub fn softmax(&self) -> Self {
        let mut mat = self.clone();
        Matrix::softmax_into(&mut mat, self);
        mat
    }

    pub fn cross_entropy(&self, other: &Self) -> Option<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut mat = Matrix::new(self.rows, self.cols);
        Matrix::cross_entropy_into(&mut mat, self, other);
        Some(mat)
    }

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

    pub fn relu_into(out: &mut Self, input: &Self) -> bool {
        if out.rows != input.rows || out.cols != input.cols {
            return false;
        }

        for i in 0..out.size() {
            out.data[i] = 0.0f32.max(input.data[i]);
        }

        true
    }

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
