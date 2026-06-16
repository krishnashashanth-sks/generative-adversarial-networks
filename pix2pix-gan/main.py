from generator import Generator
from discriminator import Discriminator
from train import fit
from optimizers import *
from losses import *
from dataset import train_dataset,test_dataset
from inference import generate_and_save_images

NUM_EPOCHS=5
OUTPUT_CHANNELS=3

generator = Generator()
generator.summary()

print("Generator model created and summarized.")

discriminator = Discriminator()
discriminator.summary()

print("Discriminator model created and summarized.")

fit(train_dataset, NUM_EPOCHS,generator,discriminator,generator_loss,discriminator_loss,generator_optimizer,discriminator_optimizer)

num_test_examples = 5
for example_input, example_target in test_dataset.take(num_test_examples):
    generate_and_save_images(generator, example_input, example_target)
