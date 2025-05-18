import onnx
from onnx.utils import Extractor
import os
from onnx import ModelProto


class MyExtractor(Extractor):
    def __init__(self, input_path: str | os.PathLike, output_path: str | os.PathLike) -> None:
        # Extractor uses infer_shapes which does not work for >2GB
        onnx.shape_inference.infer_shapes_path(input_path, output_path)
        self.model = onnx.load(output_path)
        self.graph = self.model.graph
        self.wmap = self._build_name2obj_dict(self.graph.initializer)
        self.vimap = self._build_name2obj_dict(self.graph.value_info)


def extract_model(
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
        input_names: list[str],
        output_names: list[str],
) -> None:
    onnx.checker.check_model(input_path)

    e = MyExtractor(input_path, 'model/phi/torch/model_infer.onnx')
    extracted = e.extract_model(input_names, output_names)

    onnx.save(extracted, output_path, save_as_external_data=True, all_tensors_to_one_file=False)
    onnx.checker.check_model(output_path)


def split_model(input_names, split_node_names, output_names):
    input_path = f"model/phi/torch/model.onnx"
    output_path_part1 = f"model/phi/split/p1/model.onnx"
    output_path_part2 = f"model/phi/split/p2/model.onnx"

    extract_model(
        input_path,
        output_path_part1,
        input_names +
        [f'past_key_values.{i}.key' for i in range(16)] +
        [f'past_key_values.{i}.value' for i in range(16)],
        split_node_names +
        [f'present.{i}.key' for i in range(16)] +
        [f'present.{i}.value' for i in range(16)]
    )
    extract_model(
        input_path,
        output_path_part2,
        split_node_names +
        [f'past_key_values.{i}.key' for i in range(16, 32)] +
        [f'past_key_values.{i}.value' for i in range(16, 32)],
        output_names +
        [f'present.{i}.key' for i in range(16, 32)] +
        [f'present.{i}.value' for i in range(16, 32)]
    )


if __name__ == "__main__":
    split_model(
        ["input_ids", "attention_mask", "position_ids"],
        [
            "/model/model/layers.15/Add_output_0",

            "/model/model/ScatterND_output_0",
            "/model/model/layers.14/Add_1_output_0",
            "/model/model/layers.0/self_attn/Unsqueeze_6_output_0",
            "/model/model/layers.0/self_attn/Unsqueeze_7_output_0",
        ],
        ["logits"]
    )
