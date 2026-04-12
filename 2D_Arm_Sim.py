"""
2-DOF Robot Arm Simulator
=========================
pip install pygame opencv-python numpy

python arm_sim.py                            # demo
python arm_sim.py --image photo.jpg          # trace image
python arm_sim.py --text "Hello"             # write text
python arm_sim.py --image x.jpg --port COM3  # + real arm

Keys:  O open image | T enter text | SPACE play/pause
       F frame mode | N/→ forward  | B/← back
       R restart    | L loop       | +/- speed
       S screenshot | ESC quit
       C crop/fit selector (when image loaded)
"""

import pygame
import numpy as np
import math
import argparse
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
class ArmConfig:
    L1        = 150.0
    L2        = 150.0
    S1_OFFSET = 0.0
    S2_OFFSET = 0.0
    S1_MIN, S1_MAX = 0, 180
    S2_MIN, S2_MAX = 0, 180
    CANVAS_X  = -100.0
    CANVAS_Y  =   50.0
    CANVAS_W  =  200.0
    CANVAS_H  =  150.0


# ═══════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════
class C:
    BG        = ( 13,  13,  19)
    PANEL     = ( 19,  19,  27)
    PANEL2    = ( 25,  25,  35)
    HEADER    = ( 17,  17,  25)
    GRID      = ( 30,  30,  44)
    BORDER    = ( 52,  52,  72)
    BORDER2   = ( 70,  70,  95)
    TEXT      = (208, 208, 226)
    TEXT_DIM  = ( 88,  88, 108)
    TEXT_HI   = (242, 242, 255)
    LINK1     = ( 88, 152, 255)
    LINK2     = ( 68, 212, 152)
    JOINT     = (255, 192,  68)
    PEN_DN    = (255,  72,  72)
    PEN_UP    = (108, 108, 132)
    TRACE     = (255, 255, 255)
    PATH_DIM  = ( 42,  98, 172)
    CANVAS_F  = ( 20,  20,  30)
    CANVAS_BD = ( 58,  58,  82)
    REACH     = ( 36,  36,  56)
    ACCENT    = ( 88, 178, 255)
    WARN      = (255, 172,  52)
    OK        = ( 68, 208,  88)
    BAR_BG    = ( 33,  33,  48)
    CROP_FILL = (255, 200,  50)
    CROP_BD   = (255, 230, 100)
    OVERLAY   = (  0,   0,   0)


# ═══════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════
MIN_W, MIN_H = 900, 580
HEADER_H     = 48
STATUS_H     = 28
CTRL_H       = 140


class Layout:
    def __init__(self, ww: int, wh: int):
        self.ww = ww
        self.wh = wh
        self.arm_w  = max(320, ww // 2)
        self.draw_w = ww - self.arm_w
        self.header_h  = HEADER_H
        self.content_h = wh - HEADER_H - STATUS_H

        # Arm panel
        self.shoulder  = (self.arm_w // 2,
                          HEADER_H + int(self.content_h * 0.60))
        max_px         = self.content_h * 0.42
        self.arm_scale = max_px / (ArmConfig.L1 + ArmConfig.L2)
        self.info_box  = pygame.Rect(8, HEADER_H + 8, 215, 128)
        bar_y          = HEADER_H + self.content_h - 52
        half           = (self.arm_w - 24) // 2
        self.s1_bar    = pygame.Rect(8,            bar_y, half, 12)
        self.s2_bar    = pygame.Rect(8 + half + 8, bar_y, half, 12)

        # Draw panel
        dx  = self.arm_w
        pad = 16
        bot = CTRL_H + 56
        cw  = self.draw_w - pad * 2
        ch  = self.content_h - bot - pad
        self.draw_canvas = pygame.Rect(dx + pad, HEADER_H + pad, cw, ch)
        self.scalebar_y  = self.draw_canvas.bottom + 10
        self.progress_y  = self.scalebar_y + 22
        ctrl_y           = self.progress_y + 14
        self.ctrl_box    = pygame.Rect(dx + pad, ctrl_y,
                                       self.draw_w - pad * 2, CTRL_H)
        self.status_bar  = pygame.Rect(0, wh - STATUS_H, ww, STATUS_H)


# ═══════════════════════════════════════════════════════════════════
#  KINEMATICS
# ═══════════════════════════════════════════════════════════════════
def ik(x, y, cfg, elbow_up=True):
    L1, L2 = cfg.L1, cfg.L2
    d = math.hypot(x, y)
    cos2 = max(-1.0, min(1.0, (d*d - L1*L1 - L2*L2) / (2.0*L1*L2)))
    t2   = math.acos(cos2) * (-1 if elbow_up else 1)
    t1   = math.atan2(y, x) - math.atan2(L2*math.sin(t2), L1 + L2*math.cos(t2))
    return math.degrees(t1), math.degrees(t2)


def fk(t1d, t2d, cfg):
    t1 = math.radians(t1d); t2 = math.radians(t2d)
    ex = cfg.L1*math.cos(t1); ey = cfg.L1*math.sin(t1)
    return (ex, ey), (ex + cfg.L2*math.cos(t1+t2),
                       ey + cfg.L2*math.sin(t1+t2))


# ═══════════════════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ═══════════════════════════════════════════════════════════════════
def arm_to_px(x, y, lay):
    sx, sy = lay.shoulder
    return (int(sx + x*lay.arm_scale), int(sy - y*lay.arm_scale))


def canvas_to_world(cx, cy, cfg):
    return cfg.CANVAS_X + cx, cfg.CANVAS_Y + (cfg.CANVAS_H - cy)


def world_to_draw_px(wx, wy, cfg, lay):
    r  = lay.draw_canvas
    rx = (wx - cfg.CANVAS_X) / cfg.CANVAS_W
    ry = 1.0 - (wy - cfg.CANVAS_Y) / cfg.CANVAS_H
    return (int(r.x + rx*r.width), int(r.y + ry*r.height))


# ═══════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def _sort_paths_nearest(paths: list) -> list:
    if not paths:
        return paths
    remaining = list(paths)
    ordered   = [remaining.pop(0)]
    cur_end   = ordered[0][-1]

    while remaining:
        best_i, best_d, best_rev = 0, float('inf'), False
        for i, p in enumerate(remaining):
            d_fwd = math.hypot(cur_end[0]-p[0][0],  cur_end[1]-p[0][1])
            d_rev = math.hypot(cur_end[0]-p[-1][0], cur_end[1]-p[-1][1])
            if d_fwd < best_d:
                best_i, best_d, best_rev = i, d_fwd, False
            if d_rev < best_d:
                best_i, best_d, best_rev = i, d_rev, True
        chosen = remaining.pop(best_i)
        if best_rev:
            chosen = chosen[::-1]
        ordered.append(chosen)
        cur_end = ordered[-1][-1]

    return ordered


def _simplify_path(pts: list, tol: float = 1.5) -> list:
    if len(pts) < 3:
        return pts

    def rdp(points, epsilon):
        if len(points) < 3:
            return points
        x0, y0 = points[0]
        x1, y1 = points[-1]
        dx, dy = x1-x0, y1-y0
        denom  = math.hypot(dx, dy) or 1e-9
        dists  = [abs(dy*(px-x0) - dx*(py-y0)) / denom
                  for px, py in points]
        idx    = int(np.argmax(dists))
        if dists[idx] > epsilon:
            left  = rdp(points[:idx+1], epsilon)
            right = rdp(points[idx:],   epsilon)
            return left[:-1] + right
        return [points[0], points[-1]]

    return rdp(pts, tol)


def _trace_edges(edges: np.ndarray) -> list:
    h, w    = edges.shape
    visited = np.zeros((h, w), bool)
    dirs8   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    dirs4   = [(-1,0),(0,-1),(0,1),(1,0)]

    def neighbours(y, x, d=dirs8):
        return [(y+dy, x+dx) for dy,dx in d
                if 0<=y+dy<h and 0<=x+dx<w and edges[y+dy,x+dx]]

    edge_ys, edge_xs = np.where(edges > 0)
    endpoints = []
    for y, x in zip(edge_ys, edge_xs):
        if len(neighbours(y, x)) == 1:
            endpoints.append((y, x))
    interior = [(y, x) for y, x in zip(edge_ys, edge_xs)
                if (y, x) not in set(endpoints)]
    all_starts = endpoints + interior

    paths = []
    for sy, sx in all_starts:
        if visited[sy, sx]:
            continue
        path = []
        stack = [(sy, sx, None)]
        while stack:
            y, x, prev = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            path.append((x, y))
            nbs4 = [(ny,nx) for ny,nx in neighbours(y,x,dirs4)
                    if not visited[ny,nx]]
            nbs8 = [(ny,nx) for ny,nx in neighbours(y,x)
                    if not visited[ny,nx]]
            for ny, nx in (nbs4 if nbs4 else nbs8):
                stack.append((ny, nx, (y,x)))
        if len(path) > 3:
            paths.append(path)
    return paths


def image_to_paths(image_path: str, cfg: ArmConfig,
                   crop_rect=None,
                   fit_mode: str = "fit",
                   canny_lo: int = 50,
                   canny_hi: int = 150,
                   simplify_tol: float = 3.0) -> list:
    if not HAS_CV2:
        print("[WARN] pip install opencv-python")
        return []

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERR] Cannot load: {image_path}")
        return []

    # 1. Crop
    if crop_rect is not None:
        cx, cy, cw, ch = [int(v) for v in crop_rect]
        cx = max(0, min(cx, img.shape[1]-1))
        cy = max(0, min(cy, img.shape[0]-1))
        cw = max(1, min(cw, img.shape[1]-cx))
        ch = max(1, min(ch, img.shape[0]-cy))
        img = img[cy:cy+ch, cx:cx+cw]

    ih, iw = img.shape
    canvas_ar = cfg.CANVAS_W / cfg.CANVAS_H
    image_ar  = iw / ih

    # 2. Resize respecting fit mode
    if fit_mode == "stretch":
        tw, th = 512, int(512 * cfg.CANVAS_H / cfg.CANVAS_W)
    elif fit_mode == "fill":
        if image_ar > canvas_ar:
            th = 512; tw = int(th * image_ar)
        else:
            tw = 512; th = int(tw / image_ar)
    else:
        if image_ar > canvas_ar:
            tw = 512; th = int(tw / image_ar)
        else:
            th = 512; tw = int(th * image_ar)

    tw = max(tw, 64); th = max(th, 64)
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    # 3. Contrast + blur
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4. Edge detection — no dilation, keeps edges 1px thin
    edges = cv2.Canny(img, canny_lo, canny_hi)
    try:
        edges = cv2.ximgproc.thinning(edges)
    except AttributeError:
        pass

    # 5. Trace ordered polylines
    raw = _trace_edges(edges)

    # 6. Simplify aggressively — collapses near-straight runs into
    #    single segments, eliminating redundant overlapping strokes
    simplified = [_simplify_path(p, tol=simplify_tol) for p in raw]
    # Drop zero-length stubs (start == end within 1px)
    simplified = [p for p in simplified if len(p) >= 2 and
                  math.hypot(p[-1][0]-p[0][0], p[-1][1]-p[0][1]) > 1.0]

    # 7. Scale to canvas
    if fit_mode == "fit":
        use_scale = min(cfg.CANVAS_W / tw, cfg.CANVAS_H / th)
        off_x = (cfg.CANVAS_W - tw * use_scale) / 2
        off_y = (cfg.CANVAS_H - th * use_scale) / 2
        def to_canvas(px, py):
            return (off_x + px * use_scale, off_y + py * use_scale)
    elif fit_mode == "fill":
        use_scale = max(cfg.CANVAS_W / tw, cfg.CANVAS_H / th)
        off_x = (cfg.CANVAS_W - tw * use_scale) / 2
        off_y = (cfg.CANVAS_H - th * use_scale) / 2
        def to_canvas(px, py):
            return (off_x + px * use_scale, off_y + py * use_scale)
    else:
        def to_canvas(px, py):
            return (px / tw * cfg.CANVAS_W, py / th * cfg.CANVAS_H)

    paths = []
    for p in simplified:
        cpts = [(max(0, min(cfg.CANVAS_W, to_canvas(px,py)[0])),
                 max(0, min(cfg.CANVAS_H, to_canvas(px,py)[1])))
                for px, py in p]
        if len(cpts) >= 2:
            paths.append(cpts)

    # 8. Sort to minimise pen-up travel
    paths = _sort_paths_nearest(paths)
    print(f"[IMG] {len(paths)} paths | {sum(len(p) for p in paths)} pts | fit={fit_mode}")
    return paths


# ═══════════════════════════════════════════════════════════════════
#  CROP SELECTOR
# ═══════════════════════════════════════════════════════════════════
class CropSelector:
    PANEL_W  = 700
    PANEL_H  = 520
    BTN_H    = 36
    FIT_MODES = ["fit", "fill", "stretch"]

    def __init__(self, screen, image_path, fonts, current_fit="fit"):
        self.screen     = screen
        self.image_path = image_path
        self.fonts      = fonts
        self.fit_mode   = current_fit
        self.crop_rect  = None
        self.confirmed  = False
        self.cancelled  = False

        raw = cv2.imread(image_path)
        if raw is None:
            self.cancelled = True
            return
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.src_h, self.src_w = raw.shape[:2]

        disp_w = self.PANEL_W - 20
        disp_h = self.PANEL_H - self.BTN_H * 3 - 60
        scale  = min(disp_w / self.src_w, disp_h / self.src_h)
        self.preview_scale = scale
        self.preview_w     = int(self.src_w * scale)
        self.preview_h     = int(self.src_h * scale)

        resized = cv2.resize(raw, (self.preview_w, self.preview_h),
                             interpolation=cv2.INTER_AREA)
        self.preview_surf = pygame.surfarray.make_surface(resized.swapaxes(0,1))

        self.panel_x = (screen.get_width()  - self.PANEL_W) // 2
        self.panel_y = (screen.get_height() - self.PANEL_H) // 2
        self.img_x   = self.panel_x + 10
        self.img_y   = self.panel_y + 48

        self._drag_start = None
        self._drag_cur   = None
        self._dragging   = False

        bx  = self.panel_x + 10
        by  = self.panel_y + self.PANEL_H - self.BTN_H * 2 - 18
        bw  = (self.PANEL_W - 40) // 5
        self._fit_btns = [
            (pygame.Rect(bx + i*(bw+8), by, bw, self.BTN_H), m)
            for i, m in enumerate(self.FIT_MODES)
        ]
        by2 = by + self.BTN_H + 8
        hw  = (self.PANEL_W - 36) // 2
        self._ok_btn     = pygame.Rect(bx,      by2, hw, self.BTN_H)
        self._cancel_btn = pygame.Rect(bx+hw+8, by2, hw, self.BTN_H)

    def _preview_to_src(self, px, py):
        return (max(0, min(self.src_w, int((px - self.img_x) / self.preview_scale))),
                max(0, min(self.src_h, int((py - self.img_y) / self.preview_scale))))

    def _get_norm_rect(self):
        if self._drag_start is None or self._drag_cur is None:
            return None
        x0, y0 = self._drag_start
        x1, y1 = self._drag_cur
        rx  = max(self.img_x, min(x0, x1))
        ry  = max(self.img_y, min(y0, y1))
        rx2 = min(self.img_x + self.preview_w, max(x0, x1))
        ry2 = min(self.img_y + self.preview_h, max(y0, y1))
        rw  = max(1, rx2 - rx)
        rh  = max(1, ry2 - ry)
        return rx, ry, rw, rh

    def run(self):
        clock = pygame.time.Clock()
        while not self.confirmed and not self.cancelled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cancelled = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.cancelled = True
                    if event.key == pygame.K_RETURN:
                        self._confirm()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for r, m in self._fit_btns:
                        if r.collidepoint(mx, my):
                            self.fit_mode = m
                    if self._ok_btn.collidepoint(mx, my):
                        self._confirm()
                    if self._cancel_btn.collidepoint(mx, my):
                        self.cancelled = True
                    if (self.img_x <= mx <= self.img_x + self.preview_w and
                            self.img_y <= my <= self.img_y + self.preview_h):
                        self._drag_start = (mx, my)
                        self._drag_cur   = (mx, my)
                        self._dragging   = True
                if event.type == pygame.MOUSEMOTION and self._dragging:
                    self._drag_cur = event.pos
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._dragging = False
            self._draw()
            pygame.display.flip()
            clock.tick(60)

    def _confirm(self):
        nr = self._get_norm_rect()
        if nr:
            rx, ry, rw, rh = nr
            sx, sy = self._preview_to_src(rx, ry)
            ex, ey = self._preview_to_src(rx+rw, ry+rh)
            self.crop_rect = (sx, sy, max(1, ex-sx), max(1, ey-sy))
        else:
            self.crop_rect = None
        self.confirmed = True

    def _draw(self):
        s = self.screen
        overlay = pygame.Surface(s.get_size(), pygame.SRCALPHA)
        overlay.fill((*C.OVERLAY, 180))
        s.blit(overlay, (0,0))

        px, py = self.panel_x, self.panel_y
        panel  = pygame.Surface((self.PANEL_W, self.PANEL_H), pygame.SRCALPHA)
        panel.fill((*C.PANEL2, 245))
        pygame.draw.rect(panel, (*C.BORDER2, 200), panel.get_rect(), 1, border_radius=8)
        s.blit(panel, (px, py))

        fmd = self.fonts['md']; fxs = self.fonts['xs']; fsm = self.fonts['sm']
        _txt(s, fmd, "Drag to select crop region — choose fit mode below",
             (px+10, py+12), C.TEXT_HI)

        s.blit(self.preview_surf, (self.img_x, self.img_y))
        pygame.draw.rect(s, C.BORDER2,
                         (self.img_x, self.img_y, self.preview_w, self.preview_h), 1)

        nr = self._get_norm_rect()
        if nr:
            rx, ry, rw, rh = nr
            crop_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
            crop_surf.fill((*C.CROP_FILL, 40))
            s.blit(crop_surf, (rx, ry))
            pygame.draw.rect(s, C.CROP_BD, (rx, ry, rw, rh), 2)
            sx0, sy0 = self._preview_to_src(rx, ry)
            sx1, sy1 = self._preview_to_src(rx+rw, ry+rh)
            _txt(s, fxs, f"{sx1-sx0} x {sy1-sy0} px", (rx+4, ry+4), C.CROP_BD)
        else:
            _txt(s, fxs, "No crop — full image will be used",
                 (self.img_x + self.preview_w + 10, self.img_y + 4), C.TEXT_DIM)

        for r, m in self._fit_btns:
            active = m == self.fit_mode
            pygame.draw.rect(s, C.ACCENT if active else C.BAR_BG, r, border_radius=5)
            pygame.draw.rect(s, C.TEXT_HI if active else C.BORDER, r, 1, border_radius=5)
            _txt(s, fsm, m.upper(), r.center, C.BG if active else C.TEXT_DIM, anchor="center")

        desc = {"fit": "Fit: preserves aspect, adds padding",
                "fill": "Fill: zoom to fill, may clip edges",
                "stretch": "Stretch: fills exactly, may distort"}[self.fit_mode]
        _txt(s, fxs, desc, (self.panel_x+10, self._fit_btns[0][0].bottom+4), C.TEXT_DIM)

        for r, lbl, ca, cb in [(self._ok_btn, "USE THIS", C.OK, C.BG),
                                 (self._cancel_btn, "CANCEL", C.WARN, C.BG)]:
            pygame.draw.rect(s, ca, r, border_radius=5)
            _txt(s, fmd, lbl, r.center, cb, anchor="center")


# ═══════════════════════════════════════════════════════════════════
#  TEXT → TOOLPATH
# ═══════════════════════════════════════════════════════════════════
STROKE_FONT = {
    'A':[[(0,0),(3,9),(6,0)],[(1.5,4.5),(4.5,4.5)]],'B':[[(0,0),(0,9),(3,9),(5,7.5),(3,4.5),(0,4.5)],[(3,4.5),(5,2.5),(3,0),(0,0)]],
    'C':[[(5,8),(3,9),(1,9),(0,7),(0,2),(1,0),(3,0),(5,1)]],'D':[[(0,0),(0,9),(3,9),(5,7),(5,2),(3,0),(0,0)]],
    'E':[[(5,9),(0,9),(0,0),(5,0)],[(0,4.5),(4,4.5)]],'F':[[(0,0),(0,9),(5,9)],[(0,4.5),(4,4.5)]],
    'G':[[(5,8),(3,9),(1,9),(0,7),(0,2),(1,0),(3,0),(5,1),(5,4.5),(3,4.5)]],'H':[[(0,0),(0,9)],[(6,0),(6,9)],[(0,4.5),(6,4.5)]],
    'I':[[(1,0),(5,0)],[(3,0),(3,9)],[(1,9),(5,9)]],'J':[[(0,2),(1,0),(3,0),(4,2),(4,9)],[(2,9),(6,9)]],
    'K':[[(0,0),(0,9)],[(0,4.5),(5,9)],[(0,4.5),(5,0)]],'L':[[(0,9),(0,0),(5,0)]],
    'M':[[(0,0),(0,9),(3,5),(6,9),(6,0)]],'N':[[(0,0),(0,9),(6,0),(6,9)]],
    'O':[[(1,0),(0,2),(0,7),(1,9),(3,9),(5,7),(5,2),(3,0),(1,0)]],'P':[[(0,0),(0,9),(3,9),(5,7.5),(3,4.5),(0,4.5)]],
    'Q':[[(1,0),(0,2),(0,7),(1,9),(3,9),(5,7),(5,2),(3,0),(1,0)],[(3,3),(6,0)]],'R':[[(0,0),(0,9),(3,9),(5,7.5),(3,4.5),(0,4.5)],[(3,4.5),(6,0)]],
    'S':[[(5,8),(3,9),(1,9),(0,7),(0,5),(5,4),(5,2),(4,0),(2,0),(0,1)]],'T':[[(0,9),(6,9)],[(3,9),(3,0)]],
    'U':[[(0,9),(0,2),(1,0),(3,0),(5,0),(6,2),(6,9)]],'V':[[(0,9),(3,0),(6,9)]],
    'W':[[(0,9),(1.5,0),(3,5),(4.5,0),(6,9)]],'X':[[(0,9),(6,0)],[(6,9),(0,0)]],
    'Y':[[(0,9),(3,4.5),(6,9)],[(3,4.5),(3,0)]],'Z':[[(0,9),(6,9),(0,0),(6,0)]],
    '0':[[(1,0),(0,2),(0,7),(1,9),(3,9),(5,7),(5,2),(3,0),(1,0)],[(0,2),(5,7)]],'1':[[(1,7),(3,9),(3,0)],[(1,0),(5,0)]],
    '2':[[(0,7),(1,9),(4,9),(5,7),(5,5),(0,1),(0,0),(5,0)]],'3':[[(0,8),(2,9),(4,9),(5,7),(5,5),(3,4.5),(5,3),(5,1),(4,0),(2,0),(0,1)]],
    '4':[[(0,9),(0,5),(5,5),(5,9),(5,0)]],'5':[[(5,9),(0,9),(0,5),(3,5),(5,3),(5,1),(3,0),(1,0),(0,2)]],
    '6':[[(4,9),(1,9),(0,7),(0,1),(1,0),(3,0),(5,1),(5,3),(3,5),(0,5)]],'7':[[(0,9),(5,9),(2,0)]],
    '8':[[(3,4.5),(1,4.5),(0,6),(0,8),(1,9),(3,9),(5,8),(5,6),(3,4.5),(1,4.5),(0,3),(0,1),(1,0),(3,0),(5,1),(5,3),(3,4.5)]],
    '9':[[(5,4),(2,4),(0,6),(0,8),(2,9),(4,9),(5,7),(5,1),(4,0),(2,0),(1,2)]],
    ' ':[],'!':[[(3,2),(3,9)],[(3,0),(3,1)]],'.': [[(3,0),(3,1)]],
    ',':[[(3,0),(2,-1)]],'-':[[(1,4.5),(5,4.5)]],
    '(':[[(4,9),(2,7),(2,2),(4,0)]],')': [[(2,9),(4,7),(4,2),(2,0)]],
}

def text_to_paths(text, cfg, cw=9.0, ch=13.0, sx=6.0, sy=6.0):
    paths, cx, cy = [], sx, sy
    for char in text.upper():
        if char == '\n':
            cx, cy = sx, cy + ch + 5; continue
        for stroke in STROKE_FONT.get(char, []):
            pts = [(cx + fx/6.0*cw, cy + (1.0-fy/9.0)*ch) for fx,fy in stroke]
            if pts:
                paths.append(pts)
        cx += cw + 3
        if cx + cw > cfg.CANVAS_W - 6:
            cx, cy = sx, cy + ch + 5
    return paths


# ═══════════════════════════════════════════════════════════════════
#  SERIAL
# ═══════════════════════════════════════════════════════════════════
def send_to_arm(ser, t1, t2, pen, cfg):
    """Send one command. Returns True if sent."""
    if ser is None:
        return False
    t1c = int(max(cfg.S1_MIN, min(cfg.S1_MAX, t1 + cfg.S1_OFFSET)))
    t2c = int(max(cfg.S2_MIN, min(cfg.S2_MAX, t2 + cfg.S2_OFFSET)))
    ser.write(f"{t1c},{t2c},{1 if pen else 0}\n".encode())
    return True


def poll_serial(ser) -> bool:
    """
    Non-blocking check for OK from Arduino.
    Returns True when Arduino confirms the move is done.
    Drains any other lines (debug prints etc) silently.
    """
    if ser is None or ser.in_waiting == 0:
        return False
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "OK":
            return True
        if line:
            print(f"[ARD] {line}")
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════
#  DRAW UTILS
# ═══════════════════════════════════════════════════════════════════
def _txt(surf, font, text, pos, color=C.TEXT, anchor="topleft"):
    s = font.render(text, True, color)
    surf.blit(s, s.get_rect(**{anchor: pos}))


# ═══════════════════════════════════════════════════════════════════
#  DIALOGS
# ═══════════════════════════════════════════════════════════════════
def pick_image():
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
    p = filedialog.askopenfilename(title="Open image",
        filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),("All","*.*")])
    root.destroy(); return p or None

def ask_text():
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
    t = simpledialog.askstring("Draw text", "Enter text:", parent=root)
    root.destroy(); return t or None


# ═══════════════════════════════════════════════════════════════════
#  SIMULATOR
# ═══════════════════════════════════════════════════════════════════
class ArmSim:
    def __init__(self, paths, cfg, serial_port=None,
                 image_path=None, crop_rect=None, fit_mode="fit"):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 750), pygame.RESIZABLE)
        pygame.display.set_caption("2-DOF Arm Simulator")
        self.clock = pygame.time.Clock()
        self.cfg   = cfg

        self.fxs = pygame.font.SysFont("monospace", 11)
        self.fsm = pygame.font.SysFont("monospace", 13)
        self.fmd = pygame.font.SysFont("monospace", 14, bold=True)
        self.flg = pygame.font.SysFont("monospace", 17, bold=True)
        self._fonts = {'xs':self.fxs,'sm':self.fsm,'md':self.fmd,'lg':self.flg}

        self.ser = None
        if serial_port and HAS_SERIAL:
            try:
                self.ser = serial.Serial(serial_port, 9600, timeout=1)
                print(f"[SER] {serial_port} connected")
            except Exception as e:
                print(f"[SER] {e}")

        ww, wh = self.screen.get_size()
        self.lay = Layout(ww, wh)
        self._reach_surf = None
        self._rebuild_reach()

        self._image_path = image_path
        self._crop_rect  = crop_rect
        self._fit_mode   = fit_mode

        self.playing      = False
        self.loop         = True
        self.speed        = 1
        self.step_mode    = False
        self.step_pending = 0

        self.theta1   = 45.0
        self.theta2   = -90.0
        self.pen_down = False
        self.tip_prev = None
        self.dragging = False

        # Serial handshake: True = arm ready for next command
        # When no serial connected, always True (sim runs freely)
        self._arm_ready = True

        self._toast     = ""
        self._toast_end = 0.0

        self.load_paths(paths)

    def load_paths(self, paths):
        self.all_paths   = paths
        self.flat_cmds   = self._flatten(paths)
        self.cmd_idx     = 0
        self.drawn_lines = []
        self.tip_prev    = None
        self.theta1      = 45.0
        self.theta2      = -90.0
        self.playing     = False

    def _flatten(self, paths):
        cmds = []
        for path in paths:
            for i, (cx, cy) in enumerate(path):
                wx, wy = canvas_to_world(cx, cy, self.cfg)
                cmds.append((wx, wy, i > 0))
        return cmds

    def toast(self, msg, dur=2.5):
        self._toast, self._toast_end = msg, time.time() + dur

    def on_resize(self, ww, wh):
        ww = max(ww, MIN_W); wh = max(wh, MIN_H)
        self.lay = Layout(ww, wh)
        self._rebuild_reach()

    def _rebuild_reach(self):
        lay, cfg = self.lay, self.cfg
        surf  = pygame.Surface((lay.arm_w, lay.wh), pygame.SRCALPHA)
        max_r = int((cfg.L1 + cfg.L2) * lay.arm_scale)
        min_r = int(abs(cfg.L1 - cfg.L2) * lay.arm_scale)
        pygame.draw.circle(surf, (*C.REACH, 50), lay.shoulder, max_r)
        pygame.draw.circle(surf, (*C.BG,   200), lay.shoulder, min_r)
        self._reach_surf = surf

    def move_to(self, wx, wy, pen):
        res = ik(wx, wy, self.cfg)
        if res is None: return False
        t1, t2 = res
        self.theta1, self.theta2, self.pen_down = t1, t2, pen
        tip = fk(t1, t2, self.cfg)[1]
        if pen and self.tip_prev is not None:
            self.drawn_lines.append((self.tip_prev, tip))
        self.tip_prev = tip
        if send_to_arm(self.ser, t1, t2, pen, self.cfg):
            # Mark arm as busy — wait for OK before next command
            self._arm_ready = False
        return True

    def _replay_to(self, idx):
        self.drawn_lines.clear()
        self.tip_prev = None
        self.theta1 = 45.0; self.theta2 = -90.0
        for i in range(max(0, idx)):
            self.move_to(*self.flat_cmds[i])

    def tick(self):
        # Poll Arduino for OK handshake (non-blocking, runs every frame)
        if poll_serial(self.ser):
            self._arm_ready = True

        if self.step_mode:
            if self.step_pending > 0:
                for _ in range(self.step_pending):
                    if self.cmd_idx < len(self.flat_cmds) and self._arm_ready:
                        self.move_to(*self.flat_cmds[self.cmd_idx])
                        self.cmd_idx += 1
            elif self.step_pending < 0:
                self.cmd_idx = max(0, self.cmd_idx + self.step_pending)
                self._replay_to(self.cmd_idx)
                self._arm_ready = True  # replay is sim-only, no real move sent
            self.step_pending = 0
            return

        if not self.playing: return

        # If connected to real arm, wait for OK before advancing
        if not self._arm_ready:
            return

        if self.cmd_idx >= len(self.flat_cmds):
            if self.loop:
                self.load_paths(self.all_paths); self.playing = True
            return

        # When connected: send 1 command per OK cycle (arm controls pace)
        # When sim-only: send self.speed commands per frame (fast preview)
        step_count = 1 if self.ser else self.speed
        for _ in range(step_count):
            if self.cmd_idx >= len(self.flat_cmds): break
            self.move_to(*self.flat_cmds[self.cmd_idx])
            self.cmd_idx += 1
            if self.ser:
                break  # send one, then wait for OK

    def _draw_header(self):
        s, lay = self.screen, self.lay
        pygame.draw.rect(s, C.HEADER, (0, 0, lay.ww, lay.header_h))
        pygame.draw.line(s, C.BORDER, (0, lay.header_h-1), (lay.ww, lay.header_h-1), 1)
        _txt(s, self.flg, "2-DOF Arm Simulator", (16, 14), C.TEXT_HI)
        hints = "O image  T text  C crop/fit  SPACE play  F frame  R restart  S save  ESC quit"
        _txt(s, self.fxs, hints, (lay.ww-12, 18), C.TEXT_DIM, anchor="midright")

    def _draw_status_bar(self):
        s, lay = self.screen, self.lay
        r = lay.status_bar
        pygame.draw.rect(s, C.HEADER, r)
        pygame.draw.line(s, C.BORDER, (r.x, r.y), (r.right, r.y), 1)
        total = len(self.flat_cmds)
        pct   = int(self.cmd_idx / total * 100) if total else 0
        if self.step_mode:
            stxt, scol = f"FRAME  {self.cmd_idx}/{total}", C.WARN
        elif self.playing:
            stxt, scol = f"PLAYING  {self.cmd_idx}/{total}  ({pct}%)", C.OK
        else:
            stxt, scol = f"PAUSED  {self.cmd_idx}/{total}  ({pct}%)", C.WARN
        _txt(s, self.fmd, stxt, (r.x+12, r.centery), scol, anchor="midleft")
        tip  = fk(self.theta1, self.theta2, self.cfg)[1]
        mode = f"fit={self._fit_mode}" if self._image_path else ""
        info = (f"t1={self.theta1:+.1f}  t2={self.theta2:+.1f}  "
                f"X={tip[0]:+.1f}mm  Y={tip[1]:+.1f}mm  "
                f"spd={self.speed}  loop={'on' if self.loop else 'off'}  "
                f"paths={len(self.all_paths)}  {mode}")
        _txt(s, self.fxs, info, (r.right-12, r.centery), C.TEXT_DIM, anchor="midright")

    def _draw_arm_panel(self):
        s, lay, cfg = self.screen, self.lay, self.cfg
        arm_rect = pygame.Rect(0, lay.header_h, lay.arm_w,
                               lay.wh - lay.header_h - STATUS_H)
        pygame.draw.rect(s, C.PANEL, arm_rect)
        for x in range(0, lay.arm_w, 40):
            pygame.draw.line(s, C.GRID, (x, lay.header_h), (x, arm_rect.bottom))
        for y in range(lay.header_h, arm_rect.bottom, 40):
            pygame.draw.line(s, C.GRID, (0, y), (lay.arm_w, y))
        pygame.draw.line(s, C.BORDER, (lay.arm_w, lay.header_h), (lay.arm_w, arm_rect.bottom))
        s.blit(self._reach_surf, (0, 0))

        corners = [arm_to_px(*canvas_to_world(cx, cy, cfg), lay)
                   for cx, cy in [(0,0),(cfg.CANVAS_W,0),(cfg.CANVAS_W,cfg.CANVAS_H),(0,cfg.CANVAS_H)]]
        pygame.draw.polygon(s, C.CANVAS_F,  corners)
        pygame.draw.polygon(s, C.CANVAS_BD, corners, 1)

        for a, b in self.drawn_lines[-1200:]:
            pygame.draw.line(s, C.TRACE, arm_to_px(*a, lay), arm_to_px(*b, lay), 1)

        t1r = math.radians(self.theta1); t2r = math.radians(self.theta2)
        sx, sy = lay.shoulder
        ex = int(sx + cfg.L1*math.cos(t1r)*lay.arm_scale)
        ey = int(sy - cfg.L1*math.sin(t1r)*lay.arm_scale)
        tx = int(ex + cfg.L2*math.cos(t1r+t2r)*lay.arm_scale)
        ty = int(ey - cfg.L2*math.sin(t1r+t2r)*lay.arm_scale)

        pygame.draw.line(s, (24,24,38), (sx+3,sy+3), (ex+3,ey+3), 10)
        pygame.draw.line(s, (24,24,38), (ex+3,ey+3), (tx+3,ty+3), 10)
        pygame.draw.line(s, C.LINK1, (sx,sy), (ex,ey), 7)
        pygame.draw.line(s, C.LINK2, (ex,ey), (tx,ty), 7)
        for pos, r in [((sx,sy),13), ((ex,ey),10)]:
            pygame.draw.circle(s, C.JOINT, pos, r)
            pygame.draw.circle(s, (24,24,0), pos, r, 2)
        pygame.draw.circle(s, C.PEN_DN if self.pen_down else C.PEN_UP, (tx,ty), 6)
        pygame.draw.circle(s, C.TEXT, (tx,ty), 6, 1)

        self._draw_info_box()
        self._draw_servo_bars()

    def _draw_info_box(self):
        s, lay, cfg = self.screen, self.lay, self.cfg
        r   = lay.info_box
        box = pygame.Surface(r.size, pygame.SRCALPHA)
        box.fill((*C.PANEL2, 210))
        pygame.draw.rect(box, (*C.BORDER2, 150), box.get_rect(), 1, border_radius=6)
        s.blit(box, r.topleft)
        bx, by = r.x+10, r.y
        _txt(s, self.fmd, "ARM VIEW",                    (bx, by+8),  C.TEXT_HI)
        _txt(s, self.fsm, f"t1  {self.theta1:+7.1f} deg", (bx, by+28), C.LINK1)
        _txt(s, self.fsm, f"t2  {self.theta2:+7.1f} deg", (bx, by+44), C.LINK2)
        tip = fk(self.theta1, self.theta2, cfg)[1]
        _txt(s, self.fsm, f"X   {tip[0]:+7.1f} mm",       (bx, by+62), C.TEXT)
        _txt(s, self.fsm, f"Y   {tip[1]:+7.1f} mm",       (bx, by+78), C.TEXT)
        _txt(s, self.fsm, "PEN DOWN" if self.pen_down else "PEN UP",
             (bx, by+96), C.PEN_DN if self.pen_down else C.TEXT_DIM)
        if self.step_mode:
            _txt(s, self.fsm, "[ FRAME ]", (bx, by+112), C.WARN)

    def _draw_servo_bars(self):
        s, lay = self.screen, self.lay
        def bar(rect, angle, lbl, col):
            norm = max(0.0, min(1.0, (angle % 360) / 180.0))
            pygame.draw.rect(s, C.BAR_BG, rect, border_radius=4)
            fw = max(6, int(norm * rect.width))
            pygame.draw.rect(s, col, pygame.Rect(rect.x, rect.y, fw, rect.height), border_radius=4)
            _txt(s, self.fxs, f"{lbl}  {angle:+.0f} deg", (rect.x, rect.y-15), C.TEXT_DIM)
        bar(lay.s1_bar, self.theta1,       "S1", C.LINK1)
        bar(lay.s2_bar, self.theta2 + 180, "S2", C.LINK2)

    def _draw_draw_panel(self):
        s, lay, cfg = self.screen, self.lay, self.cfg
        dx = lay.arm_w
        pygame.draw.rect(s, C.BG, (dx, lay.header_h, lay.draw_w,
                                    lay.wh - lay.header_h - STATUS_H))
        r = lay.draw_canvas
        pygame.draw.rect(s, C.CANVAS_F,  r)
        pygame.draw.rect(s, C.CANVAS_BD, r, 1)

        if self._image_path:
            badge = f"fit: {self._fit_mode}  |  press C to change"
            _txt(s, self.fxs, badge, (r.right-6, r.y+3), C.ACCENT, anchor="topright")
        _txt(s, self.fmd, "DRAW PREVIEW", (r.right-8, r.y+18), C.TEXT_HI, anchor="topright")

        for path in self.all_paths:
            pts = [world_to_draw_px(*canvas_to_world(cx, cy, cfg), cfg, lay) for cx,cy in path]
            if len(pts) > 1:
                pygame.draw.lines(s, C.PATH_DIM, False, pts, 1)

        for a, b in self.drawn_lines[-3000:]:
            pygame.draw.line(s, C.TRACE,
                             world_to_draw_px(*a, cfg, lay),
                             world_to_draw_px(*b, cfg, lay), 2)

        tip    = fk(self.theta1, self.theta2, cfg)[1]
        tip_px = world_to_draw_px(*tip, cfg, lay)
        pygame.draw.circle(s, C.PEN_DN if self.pen_down else C.PEN_UP, tip_px, 5)
        pygame.draw.circle(s, C.TEXT, tip_px, 5, 1)

        px_mm = r.width / cfg.CANVAS_W
        self._draw_scale_bar(px_mm)
        self._draw_progress_bar()
        self._draw_controls_box()
        if time.time() < self._toast_end:
            self._draw_toast()

    def _draw_scale_bar(self, px_mm):
        s, lay, cfg = self.screen, self.lay, self.cfg
        bar_mm = 50; bar_px = int(bar_mm * px_mm)
        bx, by = lay.draw_canvas.x, lay.scalebar_y
        pygame.draw.line(s, C.ACCENT, (bx, by+4), (bx+bar_px, by+4), 2)
        pygame.draw.line(s, C.ACCENT, (bx, by), (bx, by+8), 2)
        pygame.draw.line(s, C.ACCENT, (bx+bar_px, by), (bx+bar_px, by+8), 2)
        _txt(s, self.fxs, f"{bar_mm} mm", (bx+bar_px//2, by-9), C.ACCENT, anchor="midtop")
        _txt(s, self.fxs,
             f"canvas {cfg.CANVAS_W:.0f}x{cfg.CANVAS_H:.0f}mm  |  L1={cfg.L1:.0f}  L2={cfg.L2:.0f}mm",
             (bx+bar_px+12, by+1), C.TEXT_DIM)

    def _draw_progress_bar(self):
        s, lay = self.screen, self.lay
        total = len(self.flat_cmds); norm = self.cmd_idx / total if total else 0
        r = pygame.Rect(lay.draw_canvas.x, lay.progress_y, lay.draw_canvas.width, 7)
        pygame.draw.rect(s, C.BAR_BG, r, border_radius=3)
        fw = max(0, int(norm * r.width))
        if fw:
            pygame.draw.rect(s, C.ACCENT, pygame.Rect(r.x, r.y, fw, r.height), border_radius=3)

    def _draw_controls_box(self):
        s, lay = self.screen, self.lay
        r   = lay.ctrl_box
        box = pygame.Surface(r.size, pygame.SRCALPHA)
        box.fill((*C.PANEL2, 180))
        pygame.draw.rect(box, (*C.BORDER, 110), box.get_rect(), 1, border_radius=5)
        s.blit(box, r.topleft)
        bx, by = r.x+10, r.y+8
        _txt(s, self.fmd, "CONTROLS", (bx, by), C.TEXT_HI)
        rows = [
            ("O",      "open image"), ("T",      "enter text"),
            ("C",      "crop/fit"),   ("SPACE",  "play/pause"),
            ("F",      "frame mode"), ("N / ->", "step forward"),
            ("B / <-", "step back"),  ("R",      "restart"),
            ("L",      "loop"),       ("+  -",   "speed"),
        ]
        col_w = (r.width - 20) // 2
        for i, (k, d) in enumerate(rows):
            col_x = bx if i < 5 else bx + col_w
            ry    = by + 20 + (i if i < 5 else i-5) * 11
            _txt(s, self.fxs, k, (col_x,    ry), C.ACCENT)
            _txt(s, self.fxs, d, (col_x+68, ry), C.TEXT_DIM)

    def _draw_toast(self):
        s, lay = self.screen, self.lay
        r  = self.fmd.render(self._toast, True, C.TEXT_HI)
        w  = r.get_width() + 24; h = r.get_height() + 14
        bx = (lay.ww - w) // 2; by = lay.wh - STATUS_H - h - 16
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((*C.PANEL2, 235))
        pygame.draw.rect(box, (*C.ACCENT, 180), box.get_rect(), 1, border_radius=5)
        s.blit(box, (bx, by)); s.blit(r, (bx+12, by+7))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.VIDEORESIZE:
                ww = max(event.w, MIN_W); wh = max(event.h, MIN_H)
                self.screen = pygame.display.set_mode((ww,wh), pygame.RESIZABLE)
                self.on_resize(ww, wh)
            if event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_ESCAPE:  return False
                elif k == pygame.K_SPACE:   self.playing = not self.playing; self.step_mode = False
                elif k == pygame.K_r:       self.load_paths(self.all_paths); self.toast("Restarted")
                elif k == pygame.K_l:       self.loop = not self.loop; self.toast(f"Loop {'ON' if self.loop else 'OFF'}")
                elif k in (pygame.K_PLUS, pygame.K_EQUALS): self.speed = min(30, self.speed+1); self.toast(f"Speed x{self.speed}")
                elif k == pygame.K_MINUS:   self.speed = max(1, self.speed-1); self.toast(f"Speed x{self.speed}")
                elif k == pygame.K_s:
                    fn = f"arm_{int(time.time())}.png"; pygame.image.save(self.screen, fn); self.toast(f"Saved {fn}")
                elif k == pygame.K_f:       self.step_mode = not self.step_mode; self.playing = False; self.toast("Frame mode ON" if self.step_mode else "Frame mode OFF")
                elif k in (pygame.K_n, pygame.K_RIGHT) and self.step_mode: self.step_pending += 1
                elif k in (pygame.K_b, pygame.K_LEFT)  and self.step_mode: self.step_pending -= 1
                elif k == pygame.K_o: self._open_image()
                elif k == pygame.K_t: self._open_text()
                elif k == pygame.K_c: self._open_crop()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[0] < self.lay.arm_w: self.dragging = True
            if event.type == pygame.MOUSEBUTTONUP: self.dragging = False
            if event.type == pygame.MOUSEMOTION and self.dragging:
                mx, my = event.pos; sx, sy = self.lay.shoulder
                res = ik((mx-sx)/self.lay.arm_scale, (sy-my)/self.lay.arm_scale, self.cfg)
                if res: self.theta1, self.theta2 = res; self.tip_prev = fk(*res, self.cfg)[1]
        return True

    def _open_image(self, reuse_path=False):
        if not reuse_path:
            pygame.display.iconify()
            path = pick_image()
            self.screen = pygame.display.set_mode((self.lay.ww, self.lay.wh), pygame.RESIZABLE)
            if not path: self.toast("Cancelled"); return
            self._image_path = path; self._crop_rect = None; self._fit_mode = "fit"
        if not self._image_path: return
        sel = CropSelector(self.screen, self._image_path, self._fonts, self._fit_mode)
        if not sel.cancelled: sel.run()
        if sel.cancelled: self.toast("Cancelled"); return
        self._crop_rect = sel.crop_rect; self._fit_mode = sel.fit_mode
        paths = image_to_paths(self._image_path, self.cfg,
                               crop_rect=self._crop_rect, fit_mode=self._fit_mode)
        if paths:
            self.load_paths(paths)
            self.toast(f"{self._image_path.replace(chr(92),'/').split('/')[-1]}  [{self._fit_mode}]")
        else:
            self.toast("No edges found")

    def _open_text(self):
        pygame.display.iconify()
        t = ask_text()
        self.screen = pygame.display.set_mode((self.lay.ww, self.lay.wh), pygame.RESIZABLE)
        if t: self._image_path = None; self.load_paths(text_to_paths(t, self.cfg)); self.toast(f'Text: "{t}"')
        else: self.toast("Cancelled")

    def _open_crop(self):
        if not self._image_path: self.toast("No image loaded — press O first"); return
        self._open_image(reuse_path=True)

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.tick()
            self._draw_header()
            self._draw_arm_panel()
            self._draw_draw_panel()
            self._draw_status_bar()
            pygame.display.flip()
            self.clock.tick(60)
        if self.ser: self.ser.close()
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    help="Image to trace")
    parser.add_argument("--text",     help="Text to draw")
    parser.add_argument("--port",     help="Serial port e.g. COM3")
    parser.add_argument("--fit",      default="fit", choices=["fit","fill","stretch"])
    parser.add_argument("--l1",       type=float)
    parser.add_argument("--l2",       type=float)
    parser.add_argument("--canvas-w", type=float)
    parser.add_argument("--canvas-h", type=float)
    args = parser.parse_args()

    cfg = ArmConfig()
    if args.l1:       cfg.L1       = args.l1
    if args.l2:       cfg.L2       = args.l2
    if args.canvas_w: cfg.CANVAS_W = args.canvas_w
    if args.canvas_h: cfg.CANVAS_H = args.canvas_h

    if args.image:
        paths = image_to_paths(args.image, cfg, fit_mode=args.fit)
        image_path = args.image
    elif args.text:
        paths = text_to_paths(args.text, cfg)
        image_path = None
    else:
        paths = text_to_paths("HELLO", cfg)
        image_path = None

    ArmSim(paths, cfg, serial_port=args.port,
           image_path=image_path, fit_mode=args.fit).run()


if __name__ == "__main__":
    main()