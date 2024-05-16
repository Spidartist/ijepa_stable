from src.models.vision_transformer import vit_base


encoder = vit_base(img_size=[256],
        patch_size=16)

total_params = sum(p.numel() for p in encoder.parameters())
print(f"Number of parameters: {total_params}")

param_size = 0
for param in encoder.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in encoder.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('encoder size: {:.3f}MB'.format(size_all_mb))