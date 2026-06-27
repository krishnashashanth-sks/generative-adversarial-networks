import torch
from generator import BigGANGenerator
from discriminator import BigGANDiscriminator

def orthogonal_regularization(model, device=torch.device("cuda"if torch.cuda.is_available() else "cpu"),strength=1e-4, num_classes=None):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and (isinstance(model, BigGANGenerator) and 'conv' in name or isinstance(model, BigGANDiscriminator) and 'conv' in name):
            w = param.view(param.size(0), -1)
            wt_w = torch.matmul(w, w.transpose(0, 1))
            reg_loss += torch.mean((wt_w - torch.eye(wt_w.size(0), device=device))**2)

    if num_classes is not None and hasattr(model, 'label_embedding') and model.label_embedding.weight is not None:
        embed_w = model.label_embedding.weight
        if embed_w.size(0) > embed_w.size(1):
            embed_w = embed_w.transpose(0, 1)
        wt_w = torch.matmul(embed_w.transpose(0, 1), embed_w)
        reg_loss += torch.mean((wt_w - torch.eye(wt_w.size(0), device=device))**2)

    return strength * reg_loss