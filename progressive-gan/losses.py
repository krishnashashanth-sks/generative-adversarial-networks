import torch.nn as nn
import torch

class WGAN_GP_Loss(nn.Module):
  def __init__(self,lambda_gp=10,device='cpu'):
    super().__init__()
    self.lambda_gp=lambda_gp
    self.device=device
  def calculate_gradient_penalty(self,discriminator,real_images,fake_images,current_resolution):
    batch_size=real_images.size(0)
    alpha=torch.rand(batch_size,1,1,1).to(self.device)
    interpolates=(alpha*real_images+(1-alpha)*fake_images).requires_grad_(True)
    d_interpolates=discriminator(interpolates)
    gradients=torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients=gradients.view(batch_size,-1)
    gradient_norm=gradients.norm(2,1)
    gradient_penalty=((gradient_norm-1)**2).mean()
    return gradient_penalty
  def discriminator_loss(self,discriminator,real_images,fake_images,current_resolution):
    real_output=discriminator(real_images)
    fake_output=discriminator(fake_images.detach())
    gp=self.calculate_gradient_penalty(discriminator,real_images,fake_images.detach(),current_resolution)
    loss=fake_output.mean()-real_output.mean()+self.lambda_gp*gp
    return loss
  def generator_loss(self,discriminator,fake_images):
    fake_output=discriminator(fake_images)
    loss=-fake_output.mean()
    return loss
