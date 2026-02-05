use rustorch::{
    MNIST_IMG_SIZE, MNIST_LABEL_SIZE, ModelTraininingDesc, create_mnist_model, draw_mnist_digit,
    model_compile, model_create, model_feedforward, model_train, tensor::Matrix,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_images = Matrix::load(60_000, MNIST_IMG_SIZE, "mnist/train_images.mat")?;
    let test_images = Matrix::load(10_000, MNIST_IMG_SIZE, "mnist/test_images.mat")?;

    let mut train_labels = Matrix::new(60_000, MNIST_LABEL_SIZE);
    let mut test_labels = Matrix::new(10_000, MNIST_LABEL_SIZE);

    {
        let train_label_file = Matrix::load(60_000, 1, "mnist/train_labels.mat")?;
        let test_label_file = Matrix::load(10_000, 1, "mnist/test_labels.mat")?;

        for i in 0..60_000 {
            let num = train_label_file.data[i] as usize;
            train_labels.data[i * MNIST_LABEL_SIZE + num] = 1.0;
        }

        for i in 0..10_000 {
            let num = test_label_file.data[i] as usize;
            test_labels.data[i * MNIST_LABEL_SIZE + num] = 1.0;
        }
    }

    let start = fastrand::usize(0..10_000) * MNIST_IMG_SIZE;
    let end = start + MNIST_IMG_SIZE;
    draw_mnist_digit(&test_images.data[start..end]);
    // for i in 0..MNIST_LABEL_SIZE {
    //     print!("{} ", test_labels.data[i]);
    // }
    // println!("\n");

    let mut model = model_create();
    create_mnist_model(&mut model);
    model_compile(&mut model);

    let input_index = model.input.expect("Model missing input");
    if let Some(val) = model.vars[input_index].val.as_mut() {
        val.data.copy_from_slice(&test_images.data[start..end]);
    }
    model_feedforward(&mut model);

    let output_index = model.output.expect("Model missing output");
    if let Some(output) = model.vars[output_index].val.as_ref() {
        println!();
        print!("Pre training output: ");
        for i in 0..MNIST_LABEL_SIZE {
            print!("{:.2} ", output.data[i]);
        }
        println!("\n");
    }

    let training_desc = ModelTraininingDesc::new(
        train_images,
        train_labels,
        test_images,
        test_labels,
        10,
        50,
        0.01,
    );
    model_train(&mut model, &training_desc);

    let input_index = model.input.expect("Model missing input");
    let test_images = &training_desc.test_images;
    if let Some(val) = model.vars[input_index].val.as_mut() {
        val.data
            .copy_from_slice(&test_images.data[0..MNIST_IMG_SIZE]);
    }
    model_feedforward(&mut model);

    let output_index = model.output.expect("Model missing output");
    if let Some(output) = model.vars[output_index].val.as_ref() {
        println!();
        print!("Post training output: ");
        for i in 0..MNIST_LABEL_SIZE {
            print!("{} ", output.data[i]);
        }
        println!("\n");
    }

    Ok(())
}
