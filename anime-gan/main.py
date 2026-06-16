from generator import Generator
from discriminator import Discriminator, GrayDiscriminator
from utils import get_vgg_feature_extractor, 
from train import train
from optimizers import *
from losses import *
from dataset import train_dataset
from inference import infer_and_display

IMG_WIDTH = 256
IMG_HEIGHT = 256 

generator=Generator(IMG_WIDTH,IMG_HEIGHT)
generator.summary()

# Create instances of the Discriminators
discriminator = Discriminator(IMG_WIDTH,IMG_HEIGHT)
gray_discriminator = GrayDiscriminator(IMG_WIDTH,IMG_HEIGHT)

print("--- Discriminator Model Details ---")
discriminator.summary()
print(f"Discriminator expects input_image shape: {discriminator.input[0].shape}")
print(f"Discriminator expects target_image shape: {discriminator.input[1].shape}")
print(f"Discriminator output shape: {discriminator.output_shape}")

print("\n--- GrayDiscriminator Model Details ---")
gray_discriminator.summary()
print(f"GrayDiscriminator expects input_grayscale_image shape: {gray_discriminator.input.shape}")
print(f"GrayDiscriminator output shape: {gray_discriminator.output_shape}")

vgg_feature_extractor = get_vgg_feature_extractor()
print("VGG19 feature extractor loaded and ready for content and style loss calculations.")

LAMBDA_ADVERSARIAL = 10.0
LAMBDA_GRAY_ADVERSARIAL = 10.0
LAMBDA_CONTENT = 10.0
LAMBDA_STYLE = 10.0
LAMBDA_COLOR = 10.0
epochs=10

train(train_dataset, epochs,generator,discriminator,gray_discriminator,generator_adversarial_loss,gray_adversarial_loss,vgg_feature_extractor,content_loss,style_loss,color_reconstrution_loss
          ,LAMBDA_ADVERSARIAL,LAMBDA_GRAY_ADVERSARIAL,LAMBDA_CONTENT,LAMBDA_STYLE,LAMBDA_COLOR,discriminator_loss,gray_discriminator_loss,generator_optimizer,discriminator_optimizer,gray_discriminator_optimizer)

# Download a sample image (if you don't have one locally)
#!wget -O sample_photo.jpg https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616_1280.jpg

# Define paths
input_photo_path = 'sample_photo.jpg'
output_anime_path = 'generated_anime.jpg'

# Perform inference and display
infer_and_display(generator, input_photo_path,IMG_WIDTH,IMG_HEIGHT,output_anime_path)