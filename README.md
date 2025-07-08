## Usage

The worker accepts the following input parameters:

| Parameter                 | Type    | Default              | Required  | Description                                                                                                         |
| :------------------------ | :------ | :-------             | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| `prompt`                  | `str`   | `None`               | **Yes**   | The main text prompt describing the desired image.                                                                  |
| `image`                   | `str`   | `None`               | **Yes**   | Url link for image input.
| `aspect_ratio`            | `str`   | `match_input_image`  | No        | The width of the generated image in pixels                                                                          |
| `seed`                    | `int`   | `None`               | No        | Random seed for reproducibility. If `None`, a random seed is generated                                              |
| `num_inference_steps`     | `int`   | `28`                 | No        | Number of denoising steps for the base model                                                                        |
| `guidance_scale`          | `float` | `2.5`                | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt, lower values more creative       |

### Example Request

```json
{
  "input": {
    "prompt": "change the outfit to casual wear",
    "image":"https://replicate.delivery/pbxt/M2FNCc6Z6QPv8ERLIcPubmmxAFp9HnpsWi0CtK9o3ZDzchgb/Screenshot%202024-11-25%20at%201.10.30%E2%80%AFPM.png",
    "aspect_ratio": "match_input_image",
    "num_inference_steps": 28,
    "guidance_scale": 2.5,
    "seed": 589632
  }
}
```

which is producing an output like this:

```json
{
  "delayTime": 11449,
  "executionTime": 6120,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "image_url": "https://23d78952jkiodd63e921518bd08.png",
    "seed": 58964
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```
