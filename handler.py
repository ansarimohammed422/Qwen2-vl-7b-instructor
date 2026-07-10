import runpod
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import sys
import traceback

import os

CACHE_DIR = "/runpod-volume/huggingface" if os.path.exists("/runpod-volume") else None

if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Using Network Volume for Model Caching at {CACHE_DIR}")
else:
    print("No Network Volume detected. Using standard ephemeral cache.")

try:
    print("Loading Qwen2-VL Model into VRAM...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
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

        # Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

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
