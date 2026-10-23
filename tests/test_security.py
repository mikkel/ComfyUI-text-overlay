import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location("overlay",Path(__file__).parents[1]/"image_text_overlay.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_pillow13_text_rendering_and_alignments():
    node=module.ImageTextOverlay()
    image=torch.zeros(1,64,64,3)
    for alignment in ("left","center","right"):
        result=node.draw_text_on_image(image,"Test",16,32,5,"DejaVuSans.ttf",alignment,0xFFFFFF)[0]
        assert result.shape==image.shape
        assert result.sum()>0
        assert torch.equal(image,torch.zeros_like(image))
