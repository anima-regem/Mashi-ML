"""HarfBuzz shaping + unhinted FreeType rasterization, fixed em and baseline."""
import hashlib
import math
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image
from scipy.ndimage import distance_transform_edt

HEIGHT, WIDTH, EM, BASELINE, SUPERSAMPLE = 64, 96, 32, 45, 3
SDF_SCALE = 6.0


def sdf(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (HEIGHT, WIDTH) or not mask.any() or mask.all():
        raise ValueError("Expected nonempty 64x96 ink mask (True means ink)")
    return np.clip((distance_transform_edt(mask) - distance_transform_edt(~mask)) / SDF_SCALE, -1, 1).astype(np.float32)


class Renderer:
    def __init__(self, path, axes=None):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.face = freetype.Face(str(path))
        self.face.set_pixel_sizes(0, EM * SUPERSAMPLE)
        self.font = hb.Font(hb.Face(self.data))
        hb.ot_font_set_funcs(self.font)
        self.upem = self.font.face.upem
        self.font.scale = (self.upem, self.upem)
        self.axes = axes or {}
        if self.axes:
            self.font.set_variations(self.axes)
            info = self.face.get_variation_info()
            self.face.set_var_design_coords([self.axes.get(a.tag, a.default) for a in info.axes])

    def render(self, sequence):
        buf = hb.Buffer()
        buf.add_str(sequence)
        buf.direction, buf.script, buf.language = "ltr", "Mlym", "ml"
        hb.shape(self.font, buf)
        records = []
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            records.append({"glyph_id": info.codepoint, "cluster": info.cluster, "x_advance": pos.x_advance,
                            "y_advance": pos.y_advance, "x_offset": pos.x_offset, "y_offset": pos.y_offset})
        if any(g["glyph_id"] == 0 for g in records):
            return None, {"status": "notdef", "glyphs": records}
        scale = EM * SUPERSAMPLE / self.upem
        cursor_x = cursor_y = 0
        pieces = []
        for g in records:
            self.face.load_glyph(g["glyph_id"], freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
            slot = self.face.glyph
            bitmap = slot.bitmap
            if bitmap.width and bitmap.rows:
                data = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, abs(bitmap.pitch))[:, :bitmap.width]
                x = round((cursor_x + g["x_offset"]) * scale) + slot.bitmap_left
                y = BASELINE * SUPERSAMPLE - round((cursor_y + g["y_offset"]) * scale) - slot.bitmap_top
                pieces.append((x, y, data))
            cursor_x += g["x_advance"]
            cursor_y += g["y_advance"]
        if not pieces:
            return None, {"status": "empty", "glyphs": records}
        left = min(p[0] for p in pieces)
        right = max(x + a.shape[1] for x, y, a in pieces)
        top = min(p[1] for p in pieces)
        bottom = max(y + a.shape[0] for x, y, a in pieces)
        offset_x = round((WIDTH * SUPERSAMPLE - (right - left)) / 2 - left)
        if right - left > (WIDTH - 4) * SUPERSAMPLE or top < 2 * SUPERSAMPLE or bottom > (HEIGHT - 2) * SUPERSAMPLE:
            return None, {"status": "outside-canonical-canvas", "bbox_supersampled": [left, top, right, bottom], "glyphs": records}
        canvas = np.zeros((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE), np.uint8)
        for x, y, data in pieces:
            x += offset_x
            canvas[y:y + data.shape[0], x:x + data.shape[1]] = np.maximum(canvas[y:y + data.shape[0], x:x + data.shape[1]], data)
        alpha = np.asarray(Image.fromarray(canvas).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))
        mask = alpha >= 128
        if not mask.any():
            return None, {"status": "empty-after-threshold", "glyphs": records}
        return mask, {"status": "ok", "glyphs": records, "advance_em": cursor_x / self.upem,
                      "em_pixels": EM, "baseline_y": BASELINE, "horizontal_translation_pixels": offset_x / SUPERSAMPLE,
                      "native_bbox_pixels": [left / SUPERSAMPLE, top / SUPERSAMPLE, right / SUPERSAMPLE, bottom / SUPERSAMPLE],
                      "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
