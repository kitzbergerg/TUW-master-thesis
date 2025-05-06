import onnx

from optimize_model import optimize_model


def split_model(input_names, split_node_names, output_names):
    input_path = f"model/gpt2/model_torch.onnx"
    output_path_part1 = f"model/gpt2/gpt2_p1.onnx"
    output_path_part2 = f"model/gpt2/gpt2_p2.onnx"

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
        ["input_ids"],
        [
            "/transformer/h.6/Add_output_0",

            "/transformer/Expand_output_0",
            "/transformer/Concat_4_output_0"
        ],
        ["logits"]
    )
