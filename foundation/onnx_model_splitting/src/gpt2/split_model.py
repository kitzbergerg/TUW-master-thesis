import onnx

from optimize_model import optimize_model


def split_model(input_names, split_node_names, output_names):
    input_path = f"model/gpt2/model_torch.onnx"
    output_path_part1 = f"model/gpt2/gpt2_p1.onnx"
    output_path_part2 = f"model/gpt2/gpt2_p2.onnx"

    onnx.utils.extract_model(
        input_path,
        output_path_part1,
        input_names +
        [f'past_key_values.{i}.key' for i in range(6)] +
        [f'past_key_values.{i}.value' for i in range(6)],
        split_node_names +
        [f'present.{i}.key' for i in range(6)] +
        [f'present.{i}.value' for i in range(6)]
    )
    onnx.utils.extract_model(
        input_path,
        output_path_part2,
        split_node_names +
        [f'past_key_values.{i}.key' for i in range(6, 12)] +
        [f'past_key_values.{i}.value' for i in range(6, 12)],
        output_names +
        [f'present.{i}.key' for i in range(6, 12)] +
        [f'present.{i}.value' for i in range(6, 12)]
    )

    optimize_model(output_path_part1, output_path_part1)
    optimize_model(output_path_part2, output_path_part2)


if __name__ == "__main__":
    split_model(
        ["input_ids", "attention_mask", "position_ids"],
        [
            "/model/transformer/h.5/Add_output_0",

            "/model/transformer/Cast_4_output_0",
            "/model/transformer/Concat_3_output_0"
        ],
        ["logits"]
    )
