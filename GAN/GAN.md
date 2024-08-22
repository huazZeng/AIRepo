## GAN
**以手写数字为例子**
### GAN
* 判别器
  * MLP结构 最后过一层sigmoid 输出概率值
  * 通过正样本和负样本来进行训练
  * 损失函数为 **BCE**

* 生成器
  * MLP结构 最后输出图片的像素点值
  * 生成图像后，计算判别器判别结果为真的BCEloss，以此来反传调整
  ```python
  #希望结果为真，所以使用此式子计算，值越小，越真
   loss_G = criterion(outputs, real_labels) 
  ```


### condition GAN
* 与GAN相似，但是在生成器的 **输入**中加入了condition **cat** 直接拼接
  ```python
   def forward(self, x, labels):
        # 将图像和标签嵌入连接
        x = torch.cat([x, self.label_emb(labels)], dim=1)
        return self.model(x)
  ```