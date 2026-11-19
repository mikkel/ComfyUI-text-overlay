import importlib.util
from pathlib import Path
import pytest
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


def test_overlay_renders_each_batch_image_at_the_same_position():
    node = module.ImageTextOverlay()
    image = torch.zeros(2, 64, 64, 3)
    image[1].fill_(0.2)
    original = image.clone()

    result, = node.draw_text_on_image(image, "Test", 16, 32, 5, "DejaVuSans.ttf", "center", 0xFFFFFF)
    singles = [node.draw_text_on_image(frame, "Test", 16, 32, 5, "DejaVuSans.ttf", "center", 0xFFFFFF)[0]
               for frame in original]

    assert result.shape == image.shape
    assert torch.equal(result, torch.cat(singles))
    assert torch.equal(image, original)


def test_default_font_falls_back_when_arial_is_unavailable(monkeypatch):
    node = module.ImageTextOverlay()
    default_font = node.INPUT_TYPES()["required"]["font"][1]["default"]
    load_truetype = module.ImageFont.truetype

    def without_arial(font, size, *args, **kwargs):
        if font == default_font:
            raise OSError("cannot open resource")
        return load_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(module.ImageFont, "truetype", without_arial)
    image = torch.zeros(1, 64, 64, 3)
    result, = node.draw_text_on_image(image, "Test", 16, 1, 1, default_font, "left", 0xFFFFFF)

    assert result.shape == image.shape
    assert torch.count_nonzero(result) > 0


def test_missing_custom_font_still_reports_the_error(tmp_path):
    image = torch.zeros(1, 64, 64, 3)
    with pytest.raises(OSError):
        module.ImageTextOverlay().draw_text_on_image(
            image, "Test", 16, 1, 1, str(tmp_path / "missing.ttf"), "left", 0xFFFFFF
        )
