from diffusers import FluxKontextPipeline
from huggingface_hub import login
import torch
import os

def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    # HF_TOKEN 환경변수 확인 및 로그인
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("Warning: HF_TOKEN environment variable not found")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return  FluxKontextPipeline.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def get_diffusion_pipelines():
    """
    Fetches the FLUX.1-dev-kontext pipeline from the HuggingFace model hub.
    """
    pipe = fetch_pretrained_model("black-forest-labs/FLUX.1-Kontext-dev",torch_dtype=torch.bfloat16)

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()