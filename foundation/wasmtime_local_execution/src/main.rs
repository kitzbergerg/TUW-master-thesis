#![feature(array_chunks)]
use std::{fs, ops::Mul};

use wasi_nn::{
    graph::{ExecutionTarget, GraphBuilder, GraphEncoding, load},
    tensor::{Tensor, TensorType},
};

fn main() {
    let input_dim = vec![1, 3, 224, 224];
    let data = vec![
        0f32;
        input_dim
            .iter()
            .map(|&e| e as usize)
            .reduce(|acc: usize, e| acc.mul(e))
            .unwrap()
    ];

    println!(
        "{:?}",
        fs::read_dir("data")
            .unwrap()
            .map(|a| a.unwrap().path())
            .collect::<Vec<_>>()
    );

    println!("Reading file...");
    let model: GraphBuilder = fs::read("data/squeezenet1.1-7.onnx").unwrap();
    println!("Loading graph...");
    let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Gpu).unwrap();

    println!("Setting context and input...");
    let ctx = graph.init_execution_context().unwrap();
    let tensor_in = Tensor::new(
        &input_dim,
        TensorType::Fp32,
        &data.iter().map(|f32| f32.to_ne_bytes()).flatten().collect(),
    );
    ctx.set_input("data", tensor_in).unwrap();

    println!("Computing output...");
    ctx.compute().unwrap();

    println!("Parsing output...");
    let tensor_out = ctx.get_output("squeezenet0_flatten0_reshape0").unwrap();
    let output: Vec<f32> = tensor_out
        .data()
        .array_chunks()
        .map(|a| f32::from_ne_bytes(*a))
        .collect();
    println!("{:?}", output);
}
