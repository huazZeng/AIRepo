import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# 超参数设置
image_size = 28 * 28  # MNIST图像大小
hidden_dim = 256      # 隐藏层神经元数量
z_dim = 100           # 生成器输入的噪声向量维度
batch_size = 64       # 批次大小
epochs = 100          # 训练轮数
lr = 0.0002           # 学习率

# 检查和设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 图像数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size),
            nn.Tanh()  # 输出范围在[-1, 1]，与标准化的MNIST数据一致
        )
    
    def forward(self, z):
        return self.model(z)

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出一个概率值
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化生成器和判别器，并移动到GPU
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练GAN
for epoch in range(epochs):
    for batch_idx, (real_data, _) in enumerate(train_loader):
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1).to(device)
        
        # 训练判别器
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 判别器预测真实数据
        outputs_real = discriminator(real_data)
        loss_real = criterion(outputs_real, real_labels)
        
        # 生成假数据
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z)
        outputs_fake = discriminator(fake_data)
        loss_fake = criterion(outputs_fake, fake_labels)
        
        # 总判别器损失
        loss_D = loss_real + loss_fake
        
        # 更新判别器参数
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        # 训练生成器
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z)
        outputs = discriminator(fake_data)
        loss_G = criterion(outputs, real_labels)  # 希望生成的假数据被判别器认为是真实的
        
        # 更新生成器参数
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
    
    
    print(f'Epoch [{epoch}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}')
        
        # 生成图像
with torch.no_grad():
    fake_images = generator(z).view(-1, 1, 28, 28)
    fake_images = fake_images * 0.5 + 0.5  # 将数据还原到[0, 1]范围
    grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=8)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()

