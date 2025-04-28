import onnx
import onnxoptimizer
import onnxruntime as rt
from onnxruntime.transformers.onnx_model import OnnxModel


def optimize_model(input_path, output_path):
    # basic optimizations
    model = onnx.load(input_path)
    model = basic_optimizations(model)
    onnx.save(model, output_path)
    return

    # other optimizations break for web

    # optimize via offline runtime
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = output_path
    rt.InferenceSession(output_path, sess_options)


def basic_optimizations(model):
    model_optimized = onnxoptimizer.optimize(model, ["extract_constant_to_initializer", "eliminate_unused_initializer"])
    model_optimized = OnnxModel(model_optimized)
    model_optimized.prune_graph()
    remove_initializer_from_input(model_optimized.model)
    return model_optimized.model


def remove_initializer_from_input(model):
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model
