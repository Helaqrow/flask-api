from transformers import BartForConditionalGeneration, AutoTokenizer
import torch  # Required for saving as .bin

# Step 1: Load the pre-trained BART model
model_name = "facebook/bart-large-cnn"
print(f"Loading the {model_name} model...")
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Save the model and tokenizer to a directory
save_directory = "./bart_large_cnn"
print(f"Saving the model and tokenizer to '{save_directory}'...")

# Save the model weights as .bin file
torch.save(model.state_dict(), f"{save_directory}/pytorch_model.bin")  # Save weights as .bin

# Save the configuration files and tokenizer
model.config.save_pretrained(save_directory)  # Save model configuration
tokenizer.save_pretrained(save_directory)     # Save tokenizer files

print(f"Model and tokenizer saved successfully in '{save_directory}'.")
