from __future__ import annotations

import math
from typing import Callable

import pygame

from game.maps import RectObstacle
from game.world import Pickup, Projectile, World


COLORS = {
    "bg": (28, 30, 40),
    "panel": (38, 42, 56),
    "panel_border": (80, 88, 108),
    "text": (230, 232, 236),
    "subtext": (180, 186, 196),
    "learner": (90, 180, 255),
    "enemy": (255, 100, 100),
    "projectile": (255, 230, 80),
    "obstacle": (90, 95, 110),
    "pickup_health": (90, 220, 120),
    "pickup_ammo": (240, 210, 80),
    "health_bg": (90, 40, 40),
    "health_fg": (70, 210, 100),
    "aim_learner": (160, 220, 255),
    "aim_enemy": (255, 170, 170),
    "ray_wall": (160, 160, 170),
    "ray_enemy": (255, 120, 120),
    "ray_pickup": (120, 255, 150),
    "ray_projectile": (255, 235, 95),
    "ray_none": (80, 100, 120),
    "button": (55, 80, 120),
    "button_hover": (75, 105, 155),
    "button_disabled": (70, 72, 82),
    "button_text": (245, 245, 245),
    "overlay": (8, 10, 16, 180),
    "arena_boundary": (120, 130, 155),
    "nav_path": (80, 240, 120),
    "nav_path_node": (170, 255, 190),
    "desired_combat_position": (70, 255, 170),
    "navigation_waypoint": (110, 255, 110),
}


class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        caption: str,
        target_fps: int = 60,
        screen: pygame.Surface | None = None,
        show_raycast_debug: bool = False,
        show_navigation_debug: bool = True,
    ):
        pygame.init()
        if screen is None:
            existing = pygame.display.get_surface()
            if existing is None or existing.get_width() != width or existing.get_height() != height:
                self.screen = pygame.display.set_mode((width, height))
            else:
                self.screen = existing
        else:
            self.screen = screen
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps
        self.font = pygame.font.SysFont("consolas", 16)
        self.title_font = pygame.font.SysFont("consolas", 28, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.quit_requested = False
        self.back_requested = False
        self.paused = False
        self.save_requested = False
        self.show_raycast_debug = show_raycast_debug
        self.show_navigation_debug = show_navigation_debug

    def reset_runtime_flags(self) -> None:
        self.quit_requested = False
        self.back_requested = False
        self.paused = False
        self.save_requested = False

    def process_runtime_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.back_requested = True
                elif event.key == pygame.K_F5:
                    self.save_requested = True
                elif event.key == pygame.K_F3:
                    self.show_raycast_debug = not self.show_raycast_debug
                elif event.key == pygame.K_F4:
                    self.show_navigation_debug = not self.show_navigation_debug

    def sync_runtime(self, draw_callback: Callable[[], None]) -> bool:
        self.process_runtime_events()
        while self.paused and not self.quit_requested and not self.back_requested:
            self.process_runtime_events()
            draw_callback()
            self.draw_pause_overlay(["PAUSED", "Press P to resume", "F3 toggles ray debug", "F4 toggles A* path debug", "F5 saves checkpoint", "Press ESC to go back"])
            pygame.display.flip()
            self.clock.tick(30)
        return not (self.quit_requested or self.back_requested)

    def draw_world(self, world: World, hud_lines: list[str] | None = None, title: str | None = None) -> None:
        self.screen.fill(COLORS["bg"])
        self._draw_arena_boundary(world)
        self._draw_obstacles(world.obstacles)
        self._draw_pickups(world.pickups)
        if self.show_raycast_debug:
            self._draw_rays(world)
        if self.show_navigation_debug:
            self._draw_navigation_debug(world)
        self._draw_debug_points(world)
        self._draw_projectiles(world.projectiles)
        self._draw_agents(world)
        self._draw_hud(hud_lines or [], title or "")
        pygame.display.flip()
        self.clock.tick(self.target_fps)

    def draw_pause_overlay(self, lines: list[str]) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill(COLORS["overlay"])
        self.screen.blit(overlay, (0, 0))
        start_y = self.screen.get_height() // 2 - (len(lines) * 18)
        for index, line in enumerate(lines):
            font = self.title_font if index == 0 else self.font
            surface = font.render(line, True, COLORS["text"])
            x = self.screen.get_width() // 2 - surface.get_width() // 2
            y = start_y + index * 38
            self.screen.blit(surface, (x, y))

    def _draw_obstacles(self, obstacles: list[RectObstacle]) -> None:
        for obstacle in obstacles:
            rect = pygame.Rect(int(obstacle.x), int(obstacle.y), int(obstacle.w), int(obstacle.h))
            pygame.draw.rect(self.screen, COLORS["obstacle"], rect, border_radius=6)
            pygame.draw.rect(self.screen, COLORS["panel_border"], rect, 2, border_radius=6)

    def _draw_pickups(self, pickups: list[Pickup]) -> None:
        for pickup in pickups:
            if not pickup.active:
                continue
            color_key = "pickup_health" if pickup.kind == "health" else "pickup_ammo"
            pygame.draw.circle(
                self.screen,
                COLORS[color_key],
                (int(pickup.position.x), int(pickup.position.y)),
                int(pickup.radius),
            )
            label = "H" if pickup.kind == "health" else "A"
            surface = self.small_font.render(label, True, (20, 20, 20))
            self.screen.blit(
                surface,
                (
                    int(pickup.position.x - surface.get_width() / 2),
                    int(pickup.position.y - surface.get_height() / 2),
                ),
            )

    def _draw_rays(self, world: World) -> None:
        color_map = {
            0: COLORS["ray_none"],
            1: COLORS["ray_wall"],
            2: COLORS["ray_enemy"],
            3: COLORS["ray_pickup"],
            4: COLORS["ray_projectile"],
        }
        for ray in world.debug_rays:
            # New detection debug objects expose start/end/hit_point. The old
            # tuple form is still accepted for compatibility with older demos.
            if hasattr(ray, "start"):
                start, end, hit_type = ray.start, ray.end, ray.hit_type
                hit_point = ray.hit_point
            else:
                start, end, hit_type = ray
                hit_point = end if hit_type else None
            color = color_map.get(hit_type, COLORS["ray_none"])
            pygame.draw.line(self.screen, color, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 1)
            if hit_point is not None:
                pygame.draw.circle(self.screen, color, (int(hit_point[0]), int(hit_point[1])), 3)

    def _draw_navigation_debug(self, world: World) -> None:
        for polyline in getattr(world, "debug_polylines", []):
            points = [(int(x), int(y)) for x, y in polyline.points]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, COLORS["nav_path"], False, points, max(1, int(polyline.width)))
                for point in points:
                    pygame.draw.circle(self.screen, COLORS["nav_path_node"], point, 3)

    def _draw_debug_points(self, world: World) -> None:
        color_map = {
            "desired_combat_position": COLORS["desired_combat_position"],
            "navigation_waypoint": COLORS["navigation_waypoint"],
        }
        for x, y, kind in getattr(world, "debug_points", []):
            if kind in color_map:
                color = color_map[kind]
                radius = 6 if kind == "navigation_waypoint" else 5
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius, 2)
                continue
            color = COLORS["ray_wall"] if "block" in kind else COLORS["projectile"]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 4, 1)

    def consume_save_request(self) -> bool:
        requested = self.save_requested
        self.save_requested = False
        return requested

    def _draw_projectiles(self, projectiles: list[Projectile]) -> None:
        for projectile in projectiles:
            pygame.draw.circle(
                self.screen,
                COLORS["projectile"],
                (int(projectile.position.x), int(projectile.position.y)),
                int(projectile.radius + 1),
            )
            # Draw a short velocity marker so incoming projectile direction is
            # visually readable in dodging stages.
            speed = math.hypot(projectile.velocity.x, projectile.velocity.y)

            if speed > 1e-9:
                direction_x = projectile.velocity.x / speed
                direction_y = projectile.velocity.y / speed

            end_x = projectile.position.x + direction_x * 13
            end_y = projectile.position.y + direction_y * 13

            pygame.draw.line(
                self.screen,
                COLORS["projectile"],
                (int(projectile.position.x), int(projectile.position.y)),
                (int(end_x), int(end_y)),
                2,
            )

    def _draw_agents(self, world: World) -> None:
        for agent_id, agent in world.agents.items():
            if not agent.is_alive:
                continue
            color = COLORS["learner"] if agent_id == "learner" else COLORS["enemy"]
            aim_color = COLORS["aim_learner"] if agent_id == "learner" else COLORS["aim_enemy"]
            center = (int(agent.position.x), int(agent.position.y))
            pygame.draw.circle(self.screen, color, center, int(agent.radius))
            pygame.draw.circle(self.screen, (255, 255, 255), center, int(agent.radius), 2)

            # A long heading line is deliberately drawn for the RL-controlled
            # bot. Stage 1 depends almost entirely on aim direction, so the line
            # makes debugging tracking behavior much clearer than a small nose.
            aim_length = 70 if agent_id == "learner" else 50
            end_x = center[0] + math.cos(agent.heading) * aim_length
            end_y = center[1] + math.sin(agent.heading) * aim_length
            pygame.draw.line(self.screen, aim_color, center, (int(end_x), int(end_y)), 3)
            #nose_x = center[0] + math.cos(agent.heading) * agent.radius * 1.6
            #nose_y = center[1] + math.sin(agent.heading) * agent.radius * 1.6
            #pygame.draw.circle(self.screen, aim_color, (int(nose_x), int(nose_y)), 4)

            self._draw_health_bar(center[0], center[1] - 28, agent.health_ratio)
            ammo_text = self.small_font.render(f"{agent.ammo}", True, COLORS["text"])
            self.screen.blit(ammo_text, (center[0] - ammo_text.get_width() // 2, center[1] + 18))

    def _draw_health_bar(self, center_x: int, y: int, ratio: float) -> None:
        width = 40
        height = 6
        x = center_x - width // 2
        pygame.draw.rect(self.screen, COLORS["health_bg"], (x, y, width, height), border_radius=2)
        pygame.draw.rect(self.screen, COLORS["health_fg"], (x, y, int(width * ratio), height), border_radius=2)

    def _draw_hud(self, hud_lines: list[str], title: str) -> None:
        panel = pygame.Rect(10, 10, 420, max(86, 26 + 20 * len(hud_lines)))
        pygame.draw.rect(self.screen, COLORS["panel"], panel, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["panel_border"], panel, 2, border_radius=8)
        if title:
            title_surface = self.font.render(title, True, COLORS["text"])
            self.screen.blit(title_surface, (24, 20))
        for index, line in enumerate(hud_lines):
            surface = self.font.render(line, True, COLORS["text"])
            self.screen.blit(surface, (24, 44 + index * 18))

    def _draw_arena_boundary(self, world: World) -> None:
        # Stage 3 "wall avoidance" means avoiding the arena boundaries when no
        # obstacle walls are enabled, so draw the boundary explicitly.
        rect = pygame.Rect(0, 0, int(world.width) - 1, int(world.height) - 1)
        pygame.draw.rect(self.screen, COLORS["arena_boundary"], rect, 5, border_radius=2)

    def close(self) -> None:
        pygame.quit()
