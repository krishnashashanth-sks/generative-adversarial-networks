import torch.nn as nn
from layers import VectorQunatizer, Discriminator, Encoder, Decoder

class VQGAN(nn.Module):
  def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens,
               num_embeddings,embedding_dim,commitment_cost):
    super(VQGAN,self).__init__()
    self.encoder=Encoder(in_channels,num_hiddens,num_residual_layers,num_residual_hiddens)
    self.pre_quant_conv=nn.Conv2d(num_hiddens,embedding_dim,kernel_size=1)
    self.quantizer=VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
    self.post_quant_conv=nn.Conv2d(embedding_dim,num_hiddens,kernel_size=1)
    self.decoder=Decoder(in_channels,num_hiddens,num_residual_layers,num_residual_hiddens)
    self.discriminator=Discriminator(in_channels,num_hiddens)
  def forward(self,inputs):
    encoded_features=self.encoder(inputs)
    pre_quant_features=self.pre_quant_conv(encoded_features)
    quantized_latents,quant_loss,perplexity,_=self.quantizer(pre_quant_features)
    post_quant_features=self.post_quant_conv(quantized_latents)
    reconstructions=self.decoder(post_quant_features)
    return reconstructions,quant_loss,perplexity