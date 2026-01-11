use rustorch::{MNIST_IMG_DIMENTION, MNIST_IMG_SIZE, MNIST_LABEL_SIZE, Matrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_images = Matrix::load(60_000, MNIST_IMG_SIZE, "mnist/train_images.mat")?;
    let mut train_labels = Matrix::new(60_000, MNIST_LABEL_SIZE);
    train_labels.data.resize(train_labels.size(), 0f32);

    {
        let train_label_file = Matrix::load(60_000, 1, "mnist/train_labels.mat")?;
        for i in 0..60_000 {
            let num = train_label_file.data[i];
            train_labels.data[i * 10 + num as usize] = 1f32;
        }
    }

    let idx = fastrand::usize(0..60_000);
    draw_mnist_digit(&train_images.data[idx * (MNIST_IMG_SIZE)..]);
    draw_mnist_label(&train_labels.data[idx * (MNIST_LABEL_SIZE)..]);
    Ok(())
}

fn draw_mnist_digit(data: &[f32]) {
    for y in 0..MNIST_IMG_DIMENTION {
        for x in 0..MNIST_IMG_DIMENTION {
            let num = data[x + y * MNIST_IMG_DIMENTION];
            let col = 232 + (num * 24.0) as u32;
            // Here comes some ANSI magic
            print!("\x1b[48;5;{}m", col); // Set background color
            print!("  "); // Print the pixel
        }
        println!("\x1b[0m");
    }
}

fn draw_mnist_label(data: &[f32]) {
    for i in 0..MNIST_LABEL_SIZE {
        print!("{} ", data[i]);
    }
    println!()
}
