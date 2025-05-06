from transformers import AutoTokenizer, AutoModelForCausalLM

from src.test_model_output import test_model_output

if __name__ == '__main__':
    text = "Why is the sky blue?"
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model_path = "model/gpt2/model_torch.onnx"
    test_model_output(text, tokenizer, model, model_path)
