import onnx

from optimize_model import optimize_model


def split_model(model_name, split_node_names, output_names):
    input_path = f"model/{model_name}/{model_name}.onnx"
    output_path_part1 = f"model/{model_name}/{model_name}_p1.onnx"
    output_path_part2 = f"model/{model_name}/{model_name}_p2.onnx"

    model = onnx.load(input_path)

    inputs_all = set([node.name for node in model.graph.input])
    inputs_initializers = set([node.name for node in model.graph.initializer])

    input_names = list(inputs_all - inputs_initializers)

    onnx.utils.extract_model(
        input_path,
        output_path_part1,
        input_names,
        split_node_names
    )
    onnx.utils.extract_model(
        input_path,
        output_path_part2,
        split_node_names,
        output_names
    )

    optimize_model(output_path_part1, output_path_part1)
    optimize_model(output_path_part2, output_path_part2)


if __name__ == "__main__":
    split_model(
        "gpt2",
        ["/transformer/h.5/Add_output_0", "/transformer/Expand_output_0", "/transformer/Concat_4_output_0"],
        ['logits']
    )
