import numpy as np
import torch
import matplotlib.pyplot as plt
from model import VAE

# Tạo mô hình và tải trọng số
latent_dim = 50  # Đảm bảo khớp với mô hình đã huấn luyện
model = VAE(inputdim=784, hiddendim=1024, latentdim=latent_dim).cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Số lần lặp
num_samples = 8

# Tạo các đầu vào ngẫu nhiên từ phân phối chuẩn
inputs = [np.random.normal(0, 1, latent_dim).astype(np.float32) for _ in range(num_samples)]

# Chuyển đổi sang Tensor và đẩy lên GPU
inputs = [torch.from_numpy(x).unsqueeze(0).cuda() for x in inputs]  # Thêm batch dimension nếu cần

# Tạo hình ảnh từ decoder
reconstructed_images = [model.decode(x).reshape(28, 28).detach().cpu().numpy() for x in inputs]

# Hiển thị hình ảnh và giá trị đầu vào
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

for i, (ax, img, x) in enumerate(zip(axes, reconstructed_images, inputs)):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Input {i}")

plt.tight_layout()
plt.show()
