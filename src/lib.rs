pub mod model;
pub mod tensor;

pub use crate::model::{
    ModelContext, ModelTraininingDesc, ModelVar, ModelVarFlags, ModelVarId, ModelVarOperation,
    create_mnist_model, model_compile, model_create, model_feedforward, model_train,
};

pub const EPSILON: f32 = 1e-15;
pub const MNIST_IMG_DIMENTION: usize = 28;
pub const MNIST_IMG_SIZE: usize = MNIST_IMG_DIMENTION * MNIST_IMG_DIMENTION;
pub const MNIST_LABEL_SIZE: usize = 10;

pub fn draw_mnist_digit(data: &[f32]) {
    //  Render a 28x28 MNIST digit using ANSI background blocks.
    //  Uses 256-color background escape codes for a simple heatmap.
    for y in 0..MNIST_IMG_DIMENTION {
        for x in 0..MNIST_IMG_DIMENTION {
            let num = data[x + y * MNIST_IMG_DIMENTION];
            //  Map intensity [0, 1] into the 256-color grayscale ramp (232-255).
            let col = 232 + (num * 23.0) as u32;
            //  ANSI: set background to 256-color palette index and print a pixel.
            print!("\x1b[48;5;{}m  ", col);
        }
        //  ANSI: reset all attributes so the rest of the line is not colored.
        println!("\x1b[0m");
    }
}
