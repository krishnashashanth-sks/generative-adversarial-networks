import torch.nn as nn 
from IPython.utils import encoding
import torch
import torch.nn.Functional as F

class ResidualBlock(nn.Module):
  def __init__(self,num_hiddens,num_residual_hiddens):
    super(ResidualBlock,self).__init__()
    self.block=nn.Sequential(
        nn.ReLU(True),
        nn.Conv2d(in_channels=num_hiddens,
                  out_channels=num_residual_hiddens,
                  kernel_size=3,
                  stride=1,
                  padding=1,bias=False),
        nn.ReLU(True),
        nn.Conv2d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
    )
  def forward(self,x):
    return x+self.block(x)

class Encoder(nn.Module):
  def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
    super(Encoder,self).__init__()
    self.conv_1=nn.Conv2d(in_channels=in_channels,
                         out_channels=num_hiddens//2,
                         kernel_size=4,
                         stride=2,
                         padding=1)
    self.conv_2=nn.Conv2d(in_channels=num_hiddens//2,
                          out_channels=num_hiddens,
                          kernel_size=4,
                          stride=2,
                          padding=1)
    self.conv_3=nn.Conv2d(in_channels=num_hiddens,
                          out_channels=num_hiddens,
                          kernel_size=3,padding=1
                        )
    self.residual_stack=nn.Sequential(
        *[ResidualBlock(num_hiddens,num_residual_hiddens)for _ in range(num_residual_layers)]
    )
  def forward(self,inputs):
    x=self.conv_1(inputs)
    x=F.relu(x)
    x=self.conv_2(x)
    x=F.relu(x)
    x=self.conv_3(x)
    return self.residual_stack(x)

class VectorQuantizer(nn.Module):
  def __init__(self,num_embeddings,embedding_dim,commitment_cost):
    super(VectorQuantizer,self).__init__()
    self.embedding_dim=embedding_dim
    self.num_embeddings=num_embeddings
    self.commitment_cost=commitment_cost
    self.embeddings=nn.Embedding(self.num_embeddings,self.embedding_dim)
    self.embeddings.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)
  def forward(self,inputs):
    inputs=inputs.permute(0,2,3,1).contiguous()
    input_shape=inputs.shape
    flat_input=inputs.view(-1,self.embedding_dim)
    distances=(torch.sum(flat_input**2,dim=1,keepdim=True)
    +torch.sum(self.embeddings.weight**2,dim=1)
    -2*torch.matmul(flat_input,self.embeddings.weight.t()))
    encoding_indices=torch.argmin(distances,dim=1).unsqueeze(1)
    encodings=torch.zeros(encoding_indices.shape[0],self.num_embeddings,device=inputs.device)
    encodings.scatter_(1, encoding_indices, 1)
    quantized=torch.matmul(encodings,self.embeddings.weight).view(input_shape)
    e_latent_loss=F.mse_loss(quantized.detach(),inputs)
    q_latent_loss=F.mse_loss(quantized,inputs.detach())
    loss=self.commitment_cost*e_latent_loss+q_latent_loss
    quantized=inputs+(quantized-inputs).detach()
    avg_probs=torch.mean(encodings,dim=0)
    perplexity=torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))
    return quantized.permute(0,3,1,2).contiguous(),loss,perplexity,encoding_indices

class Decoder(nn.Module):
  def __init__(self,output_image_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
    super(Decoder,self).__init__()
    self.conv_1=nn.Conv2d(in_channels=num_hiddens, # Changed from in_channels to num_hiddens
                          out_channels=num_hiddens,
                          kernel_size=3,padding=1)
    self.residual_stack=nn.Sequential(
        *[ResidualBlock(num_hiddens,num_residual_hiddens)for _ in range(num_residual_layers)]
    )
    self.conv_trans_1=nn.ConvTranspose2d(in_channels=num_hiddens,
                                         out_channels=num_hiddens//2,
                                         kernel_size=4,
                                         stride=2,padding=1)
    self.conv_trans_2=nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                         out_channels=output_image_channels, # Changed from in_channels to output_image_channels
                                         kernel_size=4,stride=2,padding=1) # Changed kernel_size to 4
  def forward(self,inputs):
    # Moved F.relu to after self.conv_trans_1
    return self.conv_trans_2(F.relu(self.conv_trans_1(self.residual_stack(self.conv_1(inputs)))))

class Discriminator(nn.Module):
  def __init__(self,in_channels,num_hiddens,num_layers=3):
    super(Discriminator,self).__init__()
    layers=[]

    # First convolutional layer
    layers += [
        nn.Conv2d(in_channels, num_hiddens, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2)
    ]
    current_channels = num_hiddens

    # Subsequent convolutional layers with increasing channels
    for i in range(num_layers - 1):
      layers += [
          nn.Conv2d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(0.2)
      ]
      current_channels *= 2

    # Final layers
    layers += [
        nn.Conv2d(current_channels, current_channels, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(current_channels, 1, kernel_size=3, stride=1, padding=1)
    ]
    self.main=nn.Sequential(*layers)

  def forward(self,input):
    return self.main(input)