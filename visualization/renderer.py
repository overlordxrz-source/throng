"""
visualization/renderer.py — Beautiful dark-space renderer for THRONG v2.

Visual language:
  - Deep space background; symbol culture layer glows violet where knowledge accumulates.
  - Prey (blue team): cyan-teal glowing dots. Radius grows with brain depth.
  - Predators (red team): fiery orange-red glowing dots. Same brain-size scaling.
  - Signal lines: translucent threads between agents with similar broadcasts.
  - Death particles: small expanding rings that fade over a few frames.
  - HUD: minimal bottom-right panel — stats, brain depth, evo generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── Palette ───────────────────────────────────────────────────────────────────
BG_BASE      = (4,    6,   15)
BLUE_CORE    = (0,  230,  200)   # cyan-teal
BLUE_GLOW    = (0,   90,  220)   # deep blue
RED_CORE     = (255, 120,   20)  # orange
RED_GLOW     = (180,  20,   20)  # dark red
SIG_BLUE     = (0,  160,  200,  45)
SIG_RED      = (220,  70,   20,  45)
PARTICLE_B   = (100, 230, 255)
PARTICLE_R   = (255, 140,  60)
HUD_BG       = (8,   12,   30, 185)
HUD_BORDER   = (60,  90,  180, 140)
HUD_TEXT     = (190, 215, 255)
HUD_ACCENT   = (80,  200, 255)
FONT_SIZE    = 13


class Renderer:

    def __init__(self, window_size: int, grid_size: int) -> None:
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for visualisation")

        self.ws  = window_size
        self.gs  = grid_size
        self.cpx = window_size / grid_size

        pygame.init()
        pygame.display.set_caption("THRONG v2 — Minds Hunting Minds")
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock  = pygame.time.Clock()

        try:
            pygame.font.init()
        except Exception:
            pass
        try:
            self.font    = pygame.font.SysFont("monospace", FONT_SIZE)
            self.font_lg = pygame.font.SysFont("monospace", FONT_SIZE + 4, bold=True)
        except Exception:
            try:
                self.font    = pygame.font.Font(None, FONT_SIZE + 4)
                self.font_lg = pygame.font.Font(None, FONT_SIZE + 8)
            except Exception:
                self.font = self.font_lg = None

        # Pre-allocated surfaces
        self._bg_surf   = pygame.Surface((grid_size, grid_size))
        self._sym_surf  = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
        self._line_surf = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
        self._glow_surf = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
        self._hud_surf  = pygame.Surface((290, 148), pygame.SRCALPHA)

        # Particle system: each entry = [sx, sy, age, max_age, color_rgb]
        self._particles: List = []

    # ── Coordinate helper ─────────────────────────────────────────────────────

    def _to_screen(self, grid_pos: np.ndarray) -> Tuple[int, int]:
        return (
            int((grid_pos[1] + 0.5) * self.cpx),
            int((grid_pos[0] + 0.5) * self.cpx),
        )

    # ── Draw layers ───────────────────────────────────────────────────────────

    def _draw_background(self, symbols: np.ndarray) -> None:
        """Symbol presence → violet/indigo glow; base is deep space."""
        sym_norm = (symbols > 0).astype(np.float32)

        rgb = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        rgb[:, :, 0] = (sym_norm * 60).astype(np.uint8)    # R faint
        rgb[:, :, 1] = (sym_norm * 18).astype(np.uint8)    # G minimal
        rgb[:, :, 2] = (sym_norm * 150).astype(np.uint8)   # B dominant → violet

        # surfarray uses (x, y) = (col, row) ordering
        pygame.surfarray.blit_array(self._bg_surf, rgb.transpose(1, 0, 2))
        scaled = pygame.transform.scale(self._bg_surf, (self.ws, self.ws))

        self.screen.fill(BG_BASE)
        self.screen.blit(scaled, (0, 0), special_flags=pygame.BLEND_ADD)

    def _draw_signal_lines(
        self,
        positions:    np.ndarray,
        alive:        np.ndarray,
        team:         np.ndarray,
        signal_pairs: np.ndarray,
    ) -> None:
        if len(signal_pairs) == 0:
            return
        self._line_surf.fill((0, 0, 0, 0))
        for i, j in signal_pairs:
            if not alive[i] or not alive[j]:
                continue
            col = SIG_BLUE if int(team[i]) == 0 else SIG_RED
            pygame.draw.line(
                self._line_surf, col,
                self._to_screen(positions[i]),
                self._to_screen(positions[j]),
                1,
            )
        self.screen.blit(self._line_surf, (0, 0))

    def _draw_agents(
        self,
        positions: np.ndarray,
        alive:     np.ndarray,
        team:      np.ndarray,
        n_layers:  np.ndarray,
    ) -> None:
        """Glowing circles — radius scales with brain depth (n_layers)."""
        self._glow_surf.fill((0, 0, 0, 0))

        for idx in np.where(alive)[0]:
            pos  = positions[idx]
            t    = int(team[idx])
            nl   = max(1, int(n_layers[idx]))
            sx, sy = self._to_screen(pos)
            r    = 2 + nl   # core radius: 3–6 px

            if t == 0:
                core_c, glow_c = BLUE_CORE, BLUE_GLOW
            else:
                core_c, glow_c = RED_CORE, RED_GLOW

            # Outer halo
            pygame.draw.circle(self._glow_surf, (*glow_c, 18), (sx, sy), r + 6)
            # Mid glow
            pygame.draw.circle(self._glow_surf, (*glow_c, 55), (sx, sy), r + 3)
            # Core body
            pygame.draw.circle(self._glow_surf, (*core_c, 220), (sx, sy), r)
            # Bright centre (only for larger brains)
            if r >= 4:
                pygame.draw.circle(
                    self._glow_surf, (235, 255, 255, 190), (sx, sy), max(1, r - 2)
                )

        self.screen.blit(self._glow_surf, (0, 0))

    def _draw_particles(self) -> None:
        surviving = []
        for sx, sy, age, max_age, color in self._particles:
            frac  = 1.0 - age / max_age
            alpha = int(frac * 160)
            cur_r = 3 + age * 2
            if alpha > 8:
                try:
                    pygame.draw.circle(
                        self._glow_surf, (*color, alpha), (int(sx), int(sy)), cur_r, 1
                    )
                except Exception:
                    pass
                surviving.append([sx, sy, age + 1, max_age, color])
        self._particles = surviving

    def spawn_particle(self, grid_pos: np.ndarray, color: Tuple, n: int = 4) -> None:
        sx, sy = self._to_screen(grid_pos)
        for _ in range(n):
            self._particles.append([sx, sy, 0, 9, color])

    def _draw_hud(self, stats: Dict) -> None:
        if self.font is None:
            return

        self._hud_surf.fill(HUD_BG)

        # Title
        if self.font_lg:
            title = self.font_lg.render("THRONG v2", True, HUD_ACCENT)
            self._hud_surf.blit(title, (12, 8))

        lines = [
            f"step    {stats.get('step', 0):>9,}",
            f"blues   {stats.get('blues', 0):>5}  brain {stats.get('brain_b', '?')}",
            f"reds    {stats.get('reds',  0):>5}  brain {stats.get('brain_r', '?')}",
            f"evo gen {stats.get('evo',   0):>9}",
            f"fps     {stats.get('fps',   0.0):>9.1f}",
        ]

        y = 36
        for line in lines:
            surf = self.font.render(line, True, HUD_TEXT)
            self._hud_surf.blit(surf, (12, y))
            y += FONT_SIZE + 4

        # Border
        pygame.draw.rect(self._hud_surf, HUD_BORDER, (0, 0, 290, 148), 1)

        self.screen.blit(self._hud_surf, (self.ws - 302, self.ws - 158))

    # ── Main render entry point ───────────────────────────────────────────────

    def render(
        self,
        symbols:      np.ndarray,   # (gs, gs)
        positions:    np.ndarray,   # (max_pop, 2)
        alive:        np.ndarray,   # (max_pop,) bool
        team:         np.ndarray,   # (max_pop,) int
        n_layers:     np.ndarray,   # (max_pop,) int
        signal_pairs: np.ndarray,   # (n_pairs, 2)
        hud_stats:    Dict,
        target_fps:   int = 60,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        hud_stats["fps"] = self.clock.get_fps()

        self._draw_background(symbols)
        self._draw_signal_lines(positions, alive, team, signal_pairs)
        self._glow_surf.fill((0, 0, 0, 0))
        self._draw_agents(positions, alive, team, n_layers)
        self._draw_particles()
        self.screen.blit(self._glow_surf, (0, 0))
        self._draw_hud(hud_stats)

        pygame.display.flip()
        self.clock.tick(target_fps)
        return True

    def shutdown(self) -> None:
        pygame.quit()
