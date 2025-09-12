import os, random, gc
from dataclasses import dataclass, replace
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

# =========================
# Config
# =========================
FONTS_DIR   = "fonts"
TARGET_DIR  = "target"
OUT_DIR     = "out"
ASSETS      = "abcdefghijklmnopqrstuvwxyz1234567890~`-=][}{|';:,<.>/?!@#$%^&*()"
SEED        = None
ITERATIONS  = 200
POP_SIZE    = 100
GENS        = 20
TOPK        = 25
RAND_INJECT = 25
CHILDREN    = 2
SAVE_EVERY  = 1
VERBOSE     = True

# =========================
# Rendering
# =========================
def render_glyph(character: str, font_path: str, color: Tuple[int,int,int,int],
                 rotation: float, font_size: int) -> Image.Image:
    color = color if len(color) == 4 else (*color, 255)
    rotation = float(rotation) if rotation is not None else 0.0

    font = ImageFont.truetype(font_path, max(1, int(font_size)))
    bbox = font.getbbox(character, anchor="lt", stroke_width=0)
    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad = int(max(16, 0.5 * max(gw, gh)))
    layer = Image.new("RGBA", (gw + 2 * pad, gh + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x = pad - bbox[0]
    y = pad - bbox[1]
    draw.text((x, y), character, font=font, fill=color, anchor="lt")

    if rotation:
        layer = layer.rotate(rotation, expand=True, resample=Image.BICUBIC)

    alpha = layer.split()[3]
    tight = alpha.getbbox()
    return layer.crop(tight) if tight else Image.new("RGBA", (1, 1), (0, 0, 0, 0))

# =========================
# Helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _intersection(x0, y0, x1, y1, w, h):
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(w, x1)
    iy1 = min(h, y1)
    return ix0, iy0, ix1, iy1

def score_canvas_vs_target(canvas_rgba: Image.Image, target_lab: np.ndarray) -> float:
    c_lab = rgb2lab(np.asarray(canvas_rgba.convert("RGB"), dtype=np.float32) / 255.0)
    de = np.linalg.norm(c_lab - target_lab, axis=2).mean()
    return 100.0 - de

# =========================
# Genome
# =========================
@dataclass
class Individual:
    ch: str
    font_path: str
    color: Tuple[int, int, int, int]
    rotation: float
    font_size: int
    x: int
    y: int

def make_random_individual(assets, font_list, fonts_dir, w, h) -> Individual:
    font_path = os.path.join(fonts_dir, random.choice(font_list))
    ch = random.choice(assets)
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
    rotation = random.uniform(-25, 25)
    font_size = random.randint(100, 400)  # big allowed
    x = random.randint(-w, w - 1)
    y = random.randint(-h, h - 1)
    return Individual(ch, font_path, color, rotation, font_size, x, y)

def mutate(ind: Individual, w: int, h: int,
           pos_sigma=12, rot_sigma=6, size_sigma=6, color_sigma=18) -> Individual:
    r, g, b, a = ind.color
    r = clamp(int(r + random.gauss(0, color_sigma)), 0, 255)
    g = clamp(int(g + random.gauss(0, color_sigma)), 0, 255)
    b = clamp(int(b + random.gauss(0, color_sigma)), 0, 255)
    color = (r, g, b, a)

    rotation = ind.rotation + random.gauss(0, rot_sigma)
    font_size = max(8, int(ind.font_size + random.gauss(0, size_sigma)))

    x = int(ind.x + random.gauss(0, pos_sigma))
    y = int(ind.y + random.gauss(0, pos_sigma))

    return replace(ind, color=color, rotation=rotation, font_size=font_size, x=x, y=y)

# =========================
# Fitness
# =========================
def roi_fitness(canvas_rgba: Image.Image, canvas_lab: np.ndarray,
                target_lab: np.ndarray, ind: Individual) -> float:
    # render
    glyph = render_glyph(ind.ch, ind.font_path, ind.color, ind.rotation, ind.font_size)

    x0, y0 = ind.x, ind.y
    x1, y1 = x0 + glyph.width, y0 + glyph.height
    w, h = canvas_rgba.size
    ix0, iy0, ix1, iy1 = _intersection(x0, y0, x1, y1, w, h)

    if ix0 >= ix1 or iy0 >= iy1:
        del glyph
        return -1e6

    c_lab = canvas_lab[iy0:iy1, ix0:ix1, :]
    t_lab = target_lab[iy0:iy1, ix0:ix1, :]
    de_before = np.linalg.norm(c_lab - t_lab, axis=2).mean()

    cand_patch = canvas_rgba.crop((ix0, iy0, ix1, iy1))
    gx0 = ix0 - x0
    gy0 = iy0 - y0
    gx1 = gx0 + (ix1 - ix0)
    gy1 = gy0 + (iy1 - iy0)
    glyph_vis = glyph.crop((gx0, gy0, gx1, gy1))
    cand_patch.alpha_composite(glyph_vis, dest=(0, 0))

    cand_lab = rgb2lab(np.asarray(cand_patch.convert("RGB"), dtype=np.float32) / 255.0)
    de_after = np.linalg.norm(cand_lab - t_lab, axis=2).mean()

    del glyph, glyph_vis, cand_patch, cand_lab
    return de_before - de_after

# =========================
# Evolution
# =========================
def evolve_once(canvas_rgba: Image.Image, canvas_lab: np.ndarray, target_lab: np.ndarray,
                population: List[Individual],
                assets, font_list, fonts_dir, pop_size: int,
                gens=10, topk=25, rand_inject=25):
    w, h = canvas_rgba.size
    gen_history = []

    for g in range(gens):
        fitness = []
        for ind in population:
            score = roi_fitness(canvas_rgba, canvas_lab, target_lab, ind)
            fitness.append((score, ind))

        fitness.sort(key=lambda t: t[0], reverse=True)
        elites = [ind for _, ind in fitness[:topk]]
        best_score = fitness[0][0]
        gen_history.append((g + 1, best_score))
        if VERBOSE:
            print(f"    gen {g+1:>2}/{gens}: best={best_score:.4f}")

        next_pop: List[Individual] = elites[:]

        for _ in range(rand_inject):
            next_pop.append(make_random_individual(assets, font_list, fonts_dir, w, h))

        remaining = pop_size - len(next_pop)
        i = 0
        while remaining > 0:
            parent = elites[i % len(elites)]
            next_pop.append(mutate(parent, w, h))
            i += 1
            remaining -= 1

        population = next_pop

        del fitness, elites, next_pop
        gc.collect()

    best = max(population, key=lambda ind: roi_fitness(canvas_rgba, canvas_lab, target_lab, ind))
    return best, gen_history

# =========================
# Run
# =========================
def run(assets, fonts_dir, target_image_path,
        iterations=ITERATIONS, population_size=POP_SIZE, gens=GENS,
        save_every=SAVE_EVERY, out_dir=OUT_DIR):

    font_list = [f for f in os.listdir(fonts_dir) if f.lower().endswith((".ttf", ".otf"))]
    if not font_list:
        raise RuntimeError("No .ttf/.otf fonts found in 'fonts/'")

    target_rgba = Image.open(target_image_path).convert("RGBA")
    w, h = target_rgba.size

    target_lab = rgb2lab(np.asarray(target_rgba.convert("RGB"), dtype=np.float32) / 255.0)
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    canvas_lab = rgb2lab(np.ones_like(target_lab))

    os.makedirs(out_dir, exist_ok=True)

    for it in range(iterations):
        bar_len = 20
        pct = (it + 1) / iterations
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"[{bar}] iter {it + 1}/{iterations}")

        population = [make_random_individual(assets, font_list, fonts_dir, w, h)
                      for _ in range(population_size)]

        best, gen_hist = evolve_once(
            canvas, canvas_lab, target_lab, population,
            assets, font_list, fonts_dir, population_size,
            gens=gens, topk=TOPK, rand_inject=RAND_INJECT
        )

        glyph = render_glyph(best.ch, best.font_path, best.color, best.rotation, best.font_size)
        x0, y0 = best.x, best.y
        x1, y1 = x0 + glyph.width, y0 + glyph.height
        ix0, iy0, ix1, iy1 = _intersection(x0, y0, x1, y1, w, h)

        if ix0 < ix1 and iy0 < iy1:
            gx0 = ix0 - x0
            gy0 = iy0 - y0
            gx1 = gx0 + (ix1 - ix0)
            gy1 = gy0 + (iy1 - iy0)
            glyph_vis = glyph.crop((gx0, gy0, gx1, gy1))

            region = canvas.crop((ix0, iy0, ix1, iy1))
            region.alpha_composite(glyph_vis, dest=(0, 0))
            canvas.paste(region, (ix0, iy0))

            patch_rgb = np.asarray(region.convert("RGB"), dtype=np.float32) / 255.0
            canvas_lab[iy0:iy1, ix0:ix1, :] = rgb2lab(patch_rgb)

            del glyph_vis, region, patch_rgb
        del glyph

        iter_best = gen_hist[-1][1] if gen_hist else float("nan")
        canvas_score = score_canvas_vs_target(canvas, target_lab)
        print(f"  best of iteration (Δscore): {iter_best:.4f} | canvas score (100-ΔE76): {canvas_score:.4f}")

        if save_every and (it % save_every == 0 or it == iterations - 1):
            out_path = os.path.join(out_dir, f"step_{it:05d}.png")
            canvas.save(out_path)
            if VERBOSE:
                print(f"[iter {it}] saved {out_path}")

        del population, gen_hist
        gc.collect()

    return canvas

# =========================
# Entry
# =========================
if __name__ == "__main__":
    image_list = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_list:
        raise RuntimeError("No images found in 'target/'")
    target_path = os.path.join(TARGET_DIR, image_list[0])

    final_img = run(
        assets=ASSETS,
        fonts_dir=FONTS_DIR,
        target_image_path=target_path,
        iterations=ITERATIONS,
        population_size=POP_SIZE,
        gens=GENS,
        topk=TOPK,
        children_per_elite=CHILDREN,
        save_every=SAVE_EVERY,
        out_dir=OUT_DIR
    )
    final_img.show()
