import numpy as np
import torch
import onnxruntime as ort


def test_model_output(text, tokenizer, model, model_path):
    run_generate(text, tokenizer, model)
    print("-------------")
    run_transformers(text, tokenizer, model)
    print("-------------")
    run_onnx(text, tokenizer, model_path)


def run_generate(text, tokenizer, model):
    encoded_input = tokenizer(text, return_tensors='pt')

    generated_text = model.generate(**encoded_input)

    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(generated_text)


def run_transformers(text, tokenizer, model):
    for _ in range(0, 21):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)

        last_token = output.logits[:, -1]
        last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()
        next_token = last_token_softmax.argmax().tolist()

        text = text + tokenizer.decode(next_token)
    print(text)


def run_onnx(text, tokenizer, model_path):
    ort_sess = ort.InferenceSession(model_path)

    for _ in range(0, 21):
        encoded_input = tokenizer(text, add_special_tokens=True)['input_ids']
        input_ids = np.array([encoded_input])

        outputs = ort_sess.run(None, {'input_ids': input_ids})

        generated_ids = np.argmax(outputs[0], axis=-1)
        next_token = generated_ids[0][-1]
        text = text + tokenizer.decode(next_token)
    print(text)
