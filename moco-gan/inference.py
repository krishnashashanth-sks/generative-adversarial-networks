import torch

def generate_video_for_inference(generator_model, initial_frame, motion_sequence):
  generator_model.eval() # Set generator to evaluation mode
  with torch.no_grad(): # Disable gradient calculations
    generated_video = generator_model(initial_frame, motion_sequence)
  generator_model.train() # Set generator back to training mode (if you plan to continue training)
  return generated_video.cpu() # Move to CPU for potential visualization or saving
