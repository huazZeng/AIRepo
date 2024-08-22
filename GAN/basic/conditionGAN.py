import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

image_size = 28 * 28
hidden_dim = 256
z_dim = 100
num_classes = 10
batch_size = 64
epochs = 0
lr = 0.0002
patience = 10  # 早停的耐心次数
best_loss = float('inf')
early_stop_counter = 0

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
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_size),
            nn.Tanh()  # 输出范围在[-1, 1]，与标准化的MNIST数据一致
        )
    
    def forward(self, z, labels):
        # 将噪声和标签嵌入连接
        z = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.model(z)

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(image_size + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出一个概率值
        )
    
    def forward(self, x, labels):
        # 将图像和标签嵌入连接
        x = torch.cat([x, self.label_emb(labels)], dim=1)
        return self.model(x)

# 初始化生成器和判别器，并移动到GPU
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练cGAN并应用早停
for epoch in range(epochs):
    for batch_idx, (real_data, labels) in enumerate(train_loader):
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1).to(device)
        labels = labels.to(device)
        
        # 判别器训练
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # 判别器预测真实数据
        outputs_real = discriminator(real_data, labels)
        loss_real = criterion(outputs_real, real_labels)
        
        # 生成假数据
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z, labels)
        outputs_fake = discriminator(fake_data, labels)
        loss_fake = criterion(outputs_fake, fake_labels)
        
        # 总判别器损失
        loss_D = loss_real + loss_fake
        
        # 更新判别器参数
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        # 生成器训练
        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z, labels)
        outputs = discriminator(fake_data, labels)
        loss_G = criterion(outputs, real_labels)
        
        # 更新生成器参数
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # 打印每个 epoch 的损失
    print(f'Epoch [{epoch}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}')
    if epoch % 10 == 0:
        with torch.no_grad():
            num_images_per_class = 1
            test_labels = torch.arange(0, num_classes).repeat_interleave(num_images_per_class).to(device)
            
            # 为每个标签生成一个对应的噪声向量
            test_z = torch.randn(num_classes * num_images_per_class, z_dim).to(device)
            
            # 生成假图像
            fake_images = generator(test_z, test_labels).view(-1, 1, 28, 28)
            
            # 将数据还原到[0, 1]范围
            fake_images = fake_images * 0.5 + 0.5
            
            # 创建图像网格，nrow=5表示每行5张图片
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=5, padding=2)
            
            # 将图像张量从 (C, H, W) 转为 (H, W, C) 以适应 plt.imshow
            grid_image = grid.permute(1, 2, 0).numpy()

            # 显示图像并标注标签
            plt.figure(figsize=(15, 10))
            plt.imshow(grid_image)
            
            plt.title("Randomly Generated Images of Digits 0-9")
            plt.axis('off')  # 关闭坐标轴显示
            # 保存图像到指定路径
            plt.savefig(f"AIRepo\GAN\\basic\data\{epoch}_generated_digits.png")
            
    # 验证和早停策略
    validation_loss = loss_G.item()
    
    if validation_loss < best_loss:
        best_loss = validation_loss
        early_stop_counter = 0
        # 保存最优模型权重
        torch.save(generator.state_dict(), 'AIRepo\GAN\\basic\data\\c_best_generator.pth')
        torch.save(discriminator.state_dict(), 'AIRepo\GAN\\basic\data\\c_best_discriminator.pth')
        print(f'Saved model weights at epoch {epoch}')
    else:
        early_stop_counter += 1
    
    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# 加载保存的最佳模型权重
generator.load_state_dict(torch.load('best_generator.pth'))
discriminator.load_state_dict(torch.load('best_discriminator.pth'))    
with torch.no_grad():
    if epoch % 10 == 0:
        with torch.no_grad():
            num_images_per_class = 1
            test_labels = torch.arange(0, num_classes).repeat_interleave(num_images_per_class).to(device)
            
            # 为每个标签生成一个对应的噪声向量
            test_z = torch.randn(num_classes * num_images_per_class, z_dim).to(device)
            
            # 生成假图像
            fake_images = generator(test_z, test_labels).view(-1, 1, 28, 28)
            
            # 将数据还原到[0, 1]范围
            fake_images = fake_images * 0.5 + 0.5
            
            # 创建图像网格，nrow=5表示每行5张图片
            grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=5, padding=2)
            
            # 将图像张量从 (C, H, W) 转为 (H, W, C) 以适应 plt.imshow
            grid_image = grid.permute(1, 2, 0).numpy()

            # 显示图像并标注标签
            plt.figure(figsize=(15, 10))
            plt.imshow(grid_image)
            
            plt.title("Randomly Generated Images of Digits 0-9")
            plt.axis('off')  # 关闭坐标轴显示
            # 保存图像到指定路径
            plt.savefig(f"AIRepo\GAN\\basic\data\\final_generated_digits.png")
    

