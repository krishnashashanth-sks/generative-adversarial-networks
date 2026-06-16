# Unnormalize images (assuming normalization was to [-1, 1])
def unnormalize(img_tensor):
    img_tensor = img_tensor * 0.5 + 0.5  # Reverse normalization from (-1,1) to (0,1)
    return img_tensor