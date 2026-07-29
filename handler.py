import runpod
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import sys
import traceback

import os

# Check for RunPod's native Model Caching (The Model field in the UI)
RUNPOD_NATIVE_CACHE = "/runpod-volume/huggingface-cache"

if os.path.exists(RUNPOD_NATIVE_CACHE):
    print(f"Detected RunPod Native Model Cache at {RUNPOD_NATIVE_CACHE}")
    # Tell HuggingFace to use this pre-downloaded cache
    os.environ["HF_HOME"] = RUNPOD_NATIVE_CACHE
    # Optional: Prevent HuggingFace from pinging the internet at all since RunPod already downloaded it
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    CACHE_DIR = None # Let HF_HOME handle it automatically
elif os.path.exists("/runpod-volume"):
    print("Detected standard Network Volume. Using it for caching.")
    CACHE_DIR = "/runpod-volume/huggingface"
    os.makedirs(CACHE_DIR, exist_ok=True)
else:
    print("No caching volumes detected. Using standard ephemeral cache.")
    CACHE_DIR = None

try:
    print("Loading Qwen2-VL Model into VRAM...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa", # Fast PyTorch-native attention (no external dependencies)
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR
    )
    
    # Set reasonable limits on image resolution to prevent VRAM OOM crashes
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        cache_dir=CACHE_DIR
    )
    print("Model loaded successfully!")
except Exception as e:
    print("FATAL ERROR DURING STARTUP:", flush=True)
    traceback.print_exc()
    sys.exit(1)


def handler(job):
    """
    This runs every time your backend hits /runsync.
    The payload you send from Django is inside job["input"].
    """
    try:
        job_input = job["input"]
        messages = job_input.get("messages", [])
        
        if not messages:
            return {"error": "Invalid payload format. You must provide a 'messages' array as defined by HuggingFace's chat template structure."}

        # We expect the payload to have: {"type": "image", "image": "data:image/jpeg;base64,..."}
        # qwen_vl_utils processes this array natively!
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply the chat template (this injects the <|vision_start|> tokens perfectly)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Extract dynamic generation parameters with safe defaults
        max_new_tokens = job_input.get("max_new_tokens", 2048)
        temperature = job_input.get("temperature", 1.0)
        top_p = job_input.get("top_p", 1.0)
        
        # Determine if we should sample based on params
        do_sample = temperature != 1.0 or top_p != 1.0

        # Generate the output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )

        # Trim the prompt from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {"extracted_text": output_text[0]}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
