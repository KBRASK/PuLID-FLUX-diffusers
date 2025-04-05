from pipeline_fluxwithpulid import PuLIDFluxPipeline
import torch
from fluxforward import forward
from types import MethodType

pipe = PuLIDFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

pipe.transformer.forward = MethodType(forward, pipe.transformer)

image = pipe("holding a sign", id_image='./man.png', num_inference_steps=12, guidance_scale=3.5).images[0]

image.save("./image.png")