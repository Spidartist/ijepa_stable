import torch
from pprint import pprint

from src.models.vision_transformer import vit_base

model = vit_base(img_size=[224])
ls1 = []
ls2 = []
for k in model.state_dict():
    ls1.append(k)
ckpt = torch.load("/mnt/quanhd/ijepa_endoscopy_pretrained/mae_pretrain_vit_base.pth")
del ckpt["model"]["cls_token"]
for k in ckpt["model"]:
    ls2.append(k)


print(len(ls1))
print(len(ls2))
# print(ckpt["model"]["pos_embed"][:, 1:, :].shape)
ckpt["model"]["pos_embed"] = ckpt["model"]["pos_embed"][:, 1:, :]
print(model.load_state_dict(ckpt["model"]))