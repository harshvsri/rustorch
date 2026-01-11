#[cfg(test)]
mod tests {
    use rustorch::Matrix;

    fn create_matrix(rows: usize, cols: usize, values: &[f32]) -> Matrix {
        Matrix {
            rows,
            cols,
            data: values.to_vec(),
        }
    }

    #[test]
    fn test_transpose_rectangular() {
        // [ 1.0, 2.0, 3.0 ]
        // [ 4.0, 5.0, 6.0 ]
        let mut mat = create_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        mat.transpose();

        // [ 1.0, 4.0 ]
        // [ 2.0, 5.0 ]
        // [ 3.0, 6.0 ]
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 2);
        assert_eq!(mat.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_valid() {
        // [ 1.0, 2.0, 3.0 ]
        // [ 4.0, 5.0, 6.0 ]
        let a = create_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // [ 7.0,  8.0 ]
        // [ 9.0,  1.0 ]
        // [ 2.0,  3.0 ]
        let b = create_matrix(3, 2, &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);

        let result = a.matmul(&b);
        assert!(result.is_some());
        let res = result.unwrap();

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_matmul_mismatch() {
        let a = create_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = create_matrix(3, 3, &[1.0; 9]); // 3x3 matrix

        // 2x2 * 3x3 should fail because cols(2) != rows(3)
        let result = a.matmul(&b);
        assert!(result.is_none());
    }
}
