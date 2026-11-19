from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

DEFAULT_FONT = "arial.ttf"

class ImageTextOverlay:
    def __init__(self, device="cpu"):
        self.device = device
    _alignments = ["left", "right", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING",{"multiline": True, "default": "Hello"}),
                "font_size": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
                "font": ("STRING", {"default": DEFAULT_FONT}),
                "alignment": (cls._alignments, {"default": "left"}),  # ["left", "right", "center"]
              "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_text_on_image"
    CATEGORY = "image/text"

    def draw_text_on_image(self, image, text, font_size, x, y, font, alignment, color):
        frames = image.cpu().numpy()
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        color_rgb = (r, g, b)

        try:
            loaded_font = ImageFont.truetype(font, font_size)
        except OSError:
            if font != DEFAULT_FONT:
                raise
            loaded_font = ImageFont.load_default(size=font_size)

        rendered = np.empty(frames.shape, dtype=np.float32)
        for index, frame in enumerate(frames):
            canvas = Image.fromarray((frame * 255).astype(np.uint8))
            draw = ImageDraw.Draw(canvas)
            left, _, right, _ = draw.textbbox((0, 0), text, font=loaded_font)
            text_width = right - left
            text_x = x
            if alignment == "center":
                text_x -= text_width // 2
            elif alignment == "right":
                text_x -= text_width
            draw.text((text_x, y), text, fill=color_rgb, font=loaded_font)
            rendered[index] = np.asarray(canvas, dtype=np.float32) / 255.0

        return (torch.from_numpy(rendered),)


NODE_CLASS_MAPPINGS = {
    "Image Text Overlay": ImageTextOverlay,
}
