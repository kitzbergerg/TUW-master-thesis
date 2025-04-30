import onnx

from optimize_model import optimize_model


def split_model(split_node_names, output_names):
    input_path = f"model/gemma3/gemma-3-1b-it.onnx"
    output_path_part1 = f"model/gemma3/p1/gemma3_p1.onnx"
    output_path_part2 = f"model/gemma3/p2/gemma3_p2.onnx"

    model = onnx.load(input_path)
    print(model.graph.node)

    inputs_all = set([node.name for node in model.graph.input])
    inputs_initializers = set([node.name for node in model.graph.initializer])

    input_names = list(inputs_all - inputs_initializers)
    print(input_names)
    print(split_node_names)
    print(output_names)

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
        [
            "/model/model/layers.15/input_layernorm/Cast_output_0",
            "/model/model/layers.11/self_attn/Cast_4_output_0",
            "/model/model/layers.0/self_attn/Unsqueeze_6_output_0",
            "/model/model/layers.0/self_attn/Unsqueeze_7_output_0",
            "/model/model/layers.5/self_attn/Unsqueeze_6_output_0",
            "/model/model/layers.5/self_attn/Unsqueeze_7_output_0"
        ],
        ["logits"]
    )
