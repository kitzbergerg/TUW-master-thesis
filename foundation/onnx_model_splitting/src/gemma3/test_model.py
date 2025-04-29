from transformers import AutoTokenizer, AutoModelForCausalLM

from src.test_model_output import test_model_output

if __name__ == '__main__':
    text = "Why is the sky blue?"
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-3-1b-it')
    model_path = "model/gemma3/gemma-3-1b-it.onnx"
    test_model_output(text, tokenizer, model, model_path)
