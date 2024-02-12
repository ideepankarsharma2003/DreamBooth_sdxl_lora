# DreamBooth
DreamBooth is a powerful training technique designed to update the entire diffusion model with just a few images of a subject or style. This process is achieved by associating a special word in the prompt with example images.

Getting Started
Installation
Before running the script, make sure to install the library from the source:

```bash

git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
Navigate to the example folder with the training script and install the required dependencies:

```bash
cd examples/dreambooth
pip install -r requirements.txt
```

Accelerate Setup
Accelerate is a library for training on multiple GPUs/TPUs or with mixed-precision. Initialize an Accelerate environment:

```bash
accelerate config
```
To set up a default Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

# Train Your Model
If you want to train a model on your dataset, check the Create a dataset for training guide to learn how to create a compatible dataset for the training script.

# Training Script Parameters
DreamBooth is sensitive to training hyperparameters, so it's important to choose appropriate values. The training script provides various parameters for customization:

    --pretrained_model_name_or_path: Name of the model on the Hub or a local path to the pretrained model.
    --instance_data_dir: Path to a folder containing the training dataset (example images).
    --instance_prompt: Text prompt that contains the special word for the example images.
    --train_text_encoder: Whether to also train the text encoder.
    --output_dir: Where to save the trained model.
    --push_to_hub: Whether to push the trained model to the Hub.
    --checkpointing_steps: Frequency of saving a checkpoint as the model trains.

# Min-SNR Weighting
The Min-SNR weighting strategy can help with training by rebalancing the loss for faster convergence. Use the --snr_gamma parameter:

```bash

accelerate launch train_dreambooth.py --snr_gamma=5.0
```
# Prior Preservation Loss

Prior preservation loss uses a model’s own generated samples to help it learn to generate more diverse images. Enable it using:

```bash
accelerate launch train_dreambooth.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="path/to/class/images" \
  --class_prompt="text prompt describing class"
```

# Train Text Encoder
To improve generated outputs, train the text encoder in addition to the UNet. Enable this option if you have a GPU with at least 24GB of vRAM:

```bash
accelerate launch train_dreambooth.py --train_text_encoder
```
# Training Loop
DreamBooth comes with its own dataset classes:

    DreamBoothDataset: preprocesses the images and class images, and tokenizes the prompts for training.
    PromptDataset: generates the prompt embeddings to generate the class images.
    The training loop handles steps such as converting images to latent space, adding noise to the input, predicting the 
    noise residual, and calculating the loss.

# Launch the Script
    To launch the training script, set the necessary environment variables and execute the command:

```bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

# Inference
Once training is complete, you can use your newly trained model for inference:

```python

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("path_to_saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("A photo of sks dog in a bucket", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("dog-bucket.png")
```
# LoRA
    LoRA is a training technique for significantly reducing the number of trainable parameters. To train with LoRA, use the 
    train_dreambooth_lora.py script. Learn more in the LoRA training guide.

# Stable Diffusion XL (SDXL)
    Stable Diffusion XL (SDXL) is a text-to-image model that generates high-resolution images. Train a SDXL model with LoRA 
    using the train_dreambooth_lora_sdxl.py script. Learn more in the SDXL training guide.
