from typing import Optional, Union, Any, Tuple, List

import torch

from transformers import AutoTokenizer, AutoConfig, Gemma3ForCausalLM
from transformers.cache_utils import HybridCache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from optimum.exporters.onnx.model_configs import GemmaOnnxConfig
from optimum.exporters.onnx.convert import onnx_export_from_model


class Gemma3OnnxConfig(GemmaOnnxConfig):
    pass


class GemmaONNXWrapper(Gemma3ForCausalLM):
    """
    Wrapper around Gemma3ForCausalLM that handles past key values differently for ONNX export.
    Instead of using HybridCache objects directly, it unwraps them into tensors that can be
    traced by the ONNX exporter.
    """

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        # Convert tuple format of past_key_values to HybridCache if provided
        hybrid_cache = None
        if past_key_values is not None:
            if isinstance(past_key_values, (list, tuple)):
                # Create a new HybridCache
                hybrid_cache = HybridCache(self.config, 1, 26)

                # Unpack the tuple format:
                # past_key_values is a tuple of tuples, where the outer tuple is over layers
                # and the inner tuples are (key, value) pairs
                for layer_idx, (key, value) in enumerate(past_key_values):
                    hybrid_cache.update(key, value, layer_idx)
            else:
                # Already a HybridCache
                hybrid_cache = past_key_values

        # Call the parent class's forward method with the HybridCache
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=hybrid_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Convert HybridCache back to tuple format for ONNX compatibility
        if outputs.past_key_values is not None:
            past_cache = outputs.past_key_values

            # Convert to tuple of tuples format
            past_tuple = tuple(
                (past_cache.key_cache[i], past_cache.value_cache[i])
                for i in range(self.num_hidden_layers)
            )

            # Create a new output with the tuple format instead of HybridCache
            outputs.past_key_values = past_tuple

        return outputs


def convert_to_onnx():
    model_id = 'google/gemma-3-1b-it'
    model = GemmaONNXWrapper.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Create ONNX configurations for both models
    onnx_config = Gemma3OnnxConfig(config, task="text-generation", use_past=False)
    onnx_config_with_past = Gemma3OnnxConfig(config, task="text-generation", use_past=True)

    custom_onnx_configs = {
        "decoder_model": onnx_config,
        "decoder_with_past_model": onnx_config_with_past,
    }

    # Export both models
    onnx_export_from_model(
        model=model,
        output='model/gemma3/optimum',
        opset=18,
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=lambda model: {"decoder_model": model, "decoder_with_past_model": model},
        task="text-generation-with-past",
    )


def verify_model_outputs():
    """Verify that our wrapper produces the same outputs as the original model"""
    model_id = 'google/gemma-3-1b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = "Why is the sky blue?"

    # Original model
    print("Testing original model...")
    model_original = Gemma3ForCausalLM.from_pretrained(model_id)
    encoded_input = tokenizer(text, return_tensors='pt')

    # First pass to get initial output and past cache
    outputs_original = model_original(**encoded_input, use_cache=True)
    logits_original = outputs_original.logits

    # Our wrapper model
    print("Testing wrapper model...")
    model_wrapper = GemmaONNXWrapper.from_pretrained(model_id)

    # First pass with wrapper
    outputs_wrapper = model_wrapper(**encoded_input, use_cache=True)
    logits_wrapper = outputs_wrapper.logits

    # Compare logits
    if torch.allclose(logits_original, logits_wrapper, atol=1e-5):
        print("✅ Wrapper model outputs match original model")
    else:
        print("❌ Wrapper model outputs don't match original model")
        max_diff = torch.max(torch.abs(logits_original - logits_wrapper))
        print(f"Maximum difference: {max_diff}")

    # Generate with both models
    print("\nGenerating with original model...")
    generated_ids_original = model_original.generate(**encoded_input, max_new_tokens=20)
    generated_text_original = tokenizer.decode(generated_ids_original[0], skip_special_tokens=True)
    print(f"Original model: {generated_text_original}")

    print("\nGenerating with wrapper model...")
    generated_ids_wrapper = model_wrapper.generate(**encoded_input, max_new_tokens=20)
    generated_text_wrapper = tokenizer.decode(generated_ids_wrapper[0], skip_special_tokens=True)
    print(f"Wrapper model: {generated_text_wrapper}")

    if generated_text_original == generated_text_wrapper:
        print("✅ Generated texts match")
    else:
        print("❌ Generated texts differ")


if __name__ == "__main__":
    # First verify our wrapper works correctly
    verify_model_outputs()

    # Then convert to ONNX
    # TODO: outputs multiple model, I just want 1
    #  cache as input is missing
    convert_to_onnx()

    print("\nONNX export complete! The models should be in 'model/gemma3/optimum/'")
