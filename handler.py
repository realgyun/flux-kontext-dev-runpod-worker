import os
import base64
import json
import pprint
from PIL import Image

import torch
import runpod
from dotenv import load_dotenv
from runpod.serverless.utils import rp_download, rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import validate
from diffusers import FluxKontextPipeline

from schemas import INPUT_SCHEMA



ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (944, 1104),
    "5:4": (1104, 944),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
    "match_input_image": (None, None),
}

torch.cuda.empty_cache()



class ModelHandler:
    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        """Load the Flux Kontext pipeline from local cache."""
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", local_files_only=True
        )
        self.pipe.move_to_device("cuda")


MODELS = ModelHandler()



def get_target_size(input_image, aspect_ratio):
    """Determine target size based on the given aspect ratio."""
    if aspect_ratio == "match_input_image":
        return input_image.size

    target = ASPECT_RATIOS.get(aspect_ratio)
    if not target:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
    return target


def _save_and_upload_image(image, job_id):
    """Save a single image locally and upload to a storage bucket or encode as base64."""
    save_dir = f"/{job_id}"
    os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, "output.png")
    image.save(image_path)


    if os.environ.get("BUCKET_ENDPOINT_URL"):
        image_url = rp_upload.upload_image(job_id, image_path)
    else:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:image/png;base64,{image_data}"

    rp_cleanup.clean([save_dir])
    return image_url


@torch.inference_mode()
def generate_image(job):

    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    job_input = job["input"]
    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)


    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        print("[generate_image] Input validation error:", err, flush=True)
        raise

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}

    job_input = validated_input["validated_input"]


    if job_input.get("seed") is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")


    file_path = job_input.get("image")
    downloaded_file = rp_download.file(file_path).get("file_path")
    input_image = Image.open(downloaded_file).convert("RGB")


    target_width, target_height = get_target_size(input_image, job_input.get("aspect_ratio"))
    input_image = input_image.resize((target_width, target_height), Image.Resampling.LANCZOS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])


    try:
        result = MODELS.pipe(
            prompt=job_input["prompt"],
            image=input_image,
            height=target_height,
            width=target_width,
            num_inference_steps=job_input["num_inference_steps"],
            guidance_scale=job_input["guidance_scale"],
            generator=generator,
        )
        output_image = result.images[0]

    except RuntimeError as err:
        print(f"[ERROR] RuntimeError: {err}", flush=True)
        return {"error": f"RuntimeError: {err}", "refresh_worker": True}

    except Exception as err:
        print(f"[ERROR] Unexpected error: {err}", flush=True)
        return {"error": f"Unexpected error: {err}", "refresh_worker": True}


    image_url = _save_and_upload_image(output_image, job["id"])


    return {
        "image_url": image_url,
        "seed": job_input["seed"],
    }


runpod.serverless.start({"handler": generate_image})
