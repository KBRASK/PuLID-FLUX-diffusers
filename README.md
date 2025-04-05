# PuLID-FLUX-Diffusers

This repository integrates the **Flux version of PuLID** into the 🤗 [Diffusers](https://github.com/huggingface/diffusers) library.

With this code, you can **easily perform seamless ID-conditional image generation** using PuLID via the Diffusers pipeline.

---

## 🚀 Features

- 🔗 Full integration of PuLID (Flux version) into Diffusers
- 🧩 Seamless usage within the `DiffusionPipeline`

---

## 📦 Requirements

- `diffusers >= 0.32.0`
- Other standard libraries used by Diffusers

Install the required dependencies:

```bash
pip install diffusers>=0.32.0
```

---

## 🧑‍💻 Usage

Here is a minimal example of how to use `PuLIDFluxPipeline` for seamless ID-conditional image generation:

```python
from pipeline_fluxwithpulid import PuLIDFluxPipeline
import torch
from fluxforward import forward
from types import MethodType

# Load pipeline from pretrained FLUX model
pipe = PuLIDFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to('cuda')

# Inject custom forward function with PuLID support
pipe.transformer.forward = MethodType(forward, pipe.transformer)

# Run the pipeline with an ID image
image = pipe(
    "holding a sign",
    id_image='./man.png',
    num_inference_steps=12,
    guidance_scale=3.5
).images[0]

# Save the output
image.save("./image.png")
```
> 📥 On first use, the pipeline will automatically download the required PuLID model and EVA-CLIP weights.

Make sure your `id_image` path points to a valid reference image (e.g., face photo) and that your environment supports bfloat16 on CUDA.

---

## 📄 License

This project is released under the MIT License.

---

## 🙌 Acknowledgments

- [PuLID](https://github.com/ToTheBeginning/PuLID)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
