import torch.autograd as autograd
import torch.nn.functional as F
# --- Loss Functions ---

def d_logistic_loss(real_pred, fake_pred):
    # Standard non-saturating adversarial loss for discriminator
    real_loss = F.softplus(-real_pred) # -log(sigmoid(real_pred))
    fake_loss = F.softplus(fake_pred)  # -log(1 - sigmoid(fake_pred))
    return real_loss.mean() + fake_loss.mean()

def g_logistic_loss(fake_pred):
    # Standard non-saturating adversarial loss for generator
    return F.softplus(-fake_pred).mean()

def d_r1_loss(real_pred, real_img):
    # R1 regularization for discriminator. Encourages Lipschitz constraint.
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img,
                              create_graph=True, retain_graph=True)[0]
    grad_penalty = grad_real.pow(2).reshape(real_img.shape[0], -1).sum(1).mean()
    return grad_penalty