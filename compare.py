# from paddlenlp.transformers.mt5 import MT5Model as PDT5Model
# # from transformers.models.t5.modeling_t5 import T5Model as PTT5Model
# from transformers import MT5Model as PTT5Model
# import torch
# import paddle

# paddle.set_device("cpu")

# size = "large"
# PREFIX = "C:/Users/QYS/Desktop/"
# pt_model = PTT5Model.from_pretrained(f"{PREFIX}torch/mt5-{size}")
# pt_model.eval()
# pd_model = PDT5Model.from_pretrained("mt5-large")
# pd_model.eval()

# with paddle.no_grad():
#     pd_outputs = pd_model(
#         **pd_model.dummy_inputs,return_dict=True
#     ).last_hidden_state

# with torch.no_grad():
#     pt_outputs = pt_model(
#         **pt_model.dummy_inputs
#     ).last_hidden_state


# def compare(a, b):
#     a = torch.tensor(a.numpy()).float()
#     b = torch.tensor(b.numpy()).float()
#     meandif = (a - b).abs().mean()
#     maxdif = (a - b).abs().max()
#     print("mean difference:", meandif)
#     print("max difference:", maxdif)


# compare(pd_outputs, pt_outputs)

article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
summary = "Weiter Verhandlung in Syrien."


# torch output
import torch
import transformers
from transformers import MT5Model, T5Tokenizer

PREFIX = "C:/Users/QYS/Desktop/"
torch_model = MT5Model.from_pretrained(f"{PREFIX}torch/mt5-large")
torch_tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
torch_model.eval()
# print("111",torch_model.dummy_inputs)

torch_inputs = torch_tokenizer(article, return_tensors="pt")
with torch_tokenizer.as_target_tokenizer():
    labels = torch_tokenizer(summary, return_tensors="pt")
print("input", torch_inputs)
print("labels", labels)
torch_outputs = torch_model(input_ids=torch_inputs["input_ids"], decoder_input_ids=labels["input_ids"])
# print("output", torch_outputs)
torch_logits = torch_outputs.last_hidden_state
torch_array = torch_logits.cpu().detach().numpy()
print("torch_prediction_logits shape:{}".format(torch_array.shape))
print("torch_prediction_logits:{}".format(torch_array))


# paddle output
import paddle
import paddlenlp
from paddlenlp.transformers.mt5 import MT5Model, T5Tokenizer
import numpy as np

# paddle_model = BertForPretraining.from_pretrained(paddle_model_name)
paddle_model = MT5Model.from_pretrained("mt5-large")
paddle_tokenizer = T5Tokenizer.from_pretrained("t5-large")
paddle_model.eval()
# print("111",paddle_model.dummy_inputs)

paddle_inputs = paddle_tokenizer(article)
labels = paddle_tokenizer(summary)

paddle_inputs = {k:paddle.to_tensor([v]) for (k, v) in paddle_inputs.items()}
labels = {k:paddle.to_tensor([v]) for (k, v) in labels.items()}
print("input", paddle_inputs)
print("labels", labels)
paddle_outputs = paddle_model(input_ids=paddle_inputs["input_ids"], decoder_input_ids=labels["input_ids"],return_dict=True)
# print("output", paddle_outputs)
paddle_logits = paddle_outputs.last_hidden_state
paddle_array = paddle_logits.numpy()
print("paddle_prediction_logits shape:{}".format(paddle_array.shape))
print("paddle_prediction_logits:{}".format(paddle_array))


# the output logits should have the same shape
assert torch_array.shape == paddle_array.shape, "the output logits should have the same shape, but got : {} and {} instead".format(torch_array.shape, paddle_array.shape)
diff = torch_array - paddle_array
print(np.amax(abs(diff)))
print(np.mean(abs(diff)))
