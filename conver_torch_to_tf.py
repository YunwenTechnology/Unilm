"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/7/27 13:39
"""

from convert_unilm_pytorch_checkpoint_to_original_tf import convert_pytorch_checkpoint_to_tf
from modeling_unilm import UnilmForLM
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def f(torch_bert_dir, save_dir):
    model = UnilmForLM.from_pretrained(torch_bert_dir)
    convert_pytorch_checkpoint_to_tf(model, save_dir, "bert_model")


if __name__ == "__main__":
    torch_bert_dir = "yunwen_github/Unilm/model"
    save_dir = "yunwen_github/Unilm/model_tf"
    f(torch_bert_dir, save_dir)
