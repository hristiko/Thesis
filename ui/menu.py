from __future__ import annotations

from dataclasses import dataclass

import pygame


BACKGROUND = (24, 28, 38)
BORDER = (82, 90, 110)
TEXT = (232, 235, 240)
SUBTEXT = (190, 196, 206)
BUTTON = (60, 88, 132)
BUTTON_HOVER = (78, 112, 166)
BUTTON_DISABLED = (70, 74, 82)
WHITE = (250, 250, 250)


@dataclass
class MenuButton:
    label: str
    tag: str
    description_lines: list[str]
    enabled: bool = True
    rect: pygame.Rect | None = None


class MenuScreen:
    def __init__(self, width: int = 1500, height: int = 860, caption: str = "RL Shooter"):
        pygame.init()
        existing = pygame.display.get_surface()
        if existing is None or existing.get_width() != width or existing.get_height() != height:
            self.screen = pygame.display.set_mode((width, height))
        else:
            self.screen = existing
        pygame.display.set_caption(caption)
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("consolas", 30, bold=True)
        self.text_font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.scroll_y = 0
        self.content_height = height

    def run(
        self,
        title: str,
        subtitle: str,
        buttons: list[MenuButton],
        footer_lines: list[str] | None = None,
        allow_escape_back: bool = True,
    ) -> str | None:
        footer_lines = footer_lines or []
        self.scroll_y = 0
        self._layout_buttons(buttons)

        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and allow_escape_back:
                    return None
                if event.type == pygame.MOUSEWHEEL:
                    self.scroll_y = self._clamp_scroll(self.scroll_y - event.y * 36)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.scroll_y = self._clamp_scroll(self.scroll_y - 36)
                    elif event.button == 5:
                        self.scroll_y = self._clamp_scroll(self.scroll_y + 36)
                    elif event.button == 1:
                        scrolled_pos = (mouse_pos[0], mouse_pos[1] + self.scroll_y)
                        for button in buttons:
                            if button.enabled and button.rect and button.rect.collidepoint(scrolled_pos):
                                return button.tag

            self.screen.fill(BACKGROUND)
            self._draw_header(title, subtitle)
            self._draw_buttons(buttons, mouse_pos)
            self._draw_footer(footer_lines)
            pygame.display.flip()
            self.clock.tick(60)

    def _layout_buttons(self, buttons: list[MenuButton]) -> None:
        button_width = self.width - 120
        top = 130
        gap = 14
        current_y = top

        for button in buttons:
            line_count = max(1, len(button.description_lines))
            height = 72 + (line_count - 1) * 18
            button.rect = pygame.Rect(60, current_y, button_width, height)
            current_y += height + gap
        self.content_height = current_y + 90

    def _clamp_scroll(self, value: int) -> int:
        max_scroll = max(0, self.content_height - self.height)
        return max(0, min(max_scroll, int(value)))

    def _draw_header(self, title: str, subtitle: str) -> None:
        title_surface = self.title_font.render(title, True, TEXT)
        subtitle_surface = self.text_font.render(subtitle, True, SUBTEXT)
        self.screen.blit(title_surface, (40, 28))
        self.screen.blit(subtitle_surface, (40, 70))

    def _draw_buttons(self, buttons: list[MenuButton], mouse_pos: tuple[int, int]) -> None:
        visible_area = pygame.Rect(0, 106, self.width, self.height - 160)
        for button in buttons:
            if button.rect is None:
                continue

            rect = button.rect.move(0, -self.scroll_y)
            if rect.bottom < visible_area.top or rect.top > visible_area.bottom:
                continue

            hovered = rect.collidepoint(mouse_pos)
            if not button.enabled:
                color = BUTTON_DISABLED
            elif hovered:
                color = BUTTON_HOVER
            else:
                color = BUTTON

            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, BORDER, rect, 2, border_radius=8)

            label_surface = self.text_font.render(button.label, True, WHITE if button.enabled else SUBTEXT)
            self.screen.blit(label_surface, (rect.x + 16, rect.y + 12))

            for line_index, line in enumerate(button.description_lines):
                line_surface = self.small_font.render(line, True, TEXT if button.enabled else SUBTEXT)
                self.screen.blit(line_surface, (rect.x + 16, rect.y + 40 + line_index * 18))

    def _draw_footer(self, footer_lines: list[str]) -> None:
        base_y = self.height - 24 - len(footer_lines) * 18
        for index, line in enumerate(footer_lines):
            surface = self.small_font.render(line, True, SUBTEXT)
            self.screen.blit(surface, (40, base_y + index * 18))


def show_main_menu() -> str:
    menu = MenuScreen(caption="RL Shooter - Main Menu")
    buttons = [
        MenuButton("Play human vs stock bot", "play_stock", ["Human-controlled player against the improved scripted stock bot."]),
        MenuButton("Train MaskablePPO bot", "train", ["Choose a curriculum stage. Headless training is the default; visual training is optional."]),
        MenuButton("Watch / test trained RL bot", "test", ["Choose a saved validated MaskablePPO/PPO bot and play against it."]),
        MenuButton("Run Stage 0 sanity check", "sanity", ["Scripted validation of movement, shooting, collisions, pickups, rewards, observations, and reset logic."]),
        MenuButton("Quit", "quit", ["Close the game."]),
    ]
    result = menu.run(
        title="Standalone Top-Down 2D Shooter",
        subtitle="RL curriculum thesis project",
        buttons=buttons,
        footer_lines=["Mouse: menu click | Mouse wheel: scroll", "ESC: back in submenus | P: pause in gameplay/training | F3: ray debug | F5: save"],
        allow_escape_back=False,
    )
    return result or "quit"


def show_train_menu(stage_rows: list[dict], last_trained_label: str) -> str | None:
    menu = MenuScreen(caption="RL Shooter - Train")
    buttons: list[MenuButton] = []

    for row in stage_rows:
        if not row.get("trainable", True):
            buttons.append(
                MenuButton(
                    label=row["name"],
                    tag=row["stage_id"],
                    enabled=True,
                    description_lines=[
                        row["description"],
                        "This is a scripted sanity-check stage, not PPO/MaskablePPO training.",
                        f"Passed: {row.get('passed_validation', False)}",
                    ],
                )
            )
            continue

        buttons.append(
            MenuButton(
                label=row["name"],
                tag=row["stage_id"],
                enabled=bool(row["unlocked"]),
                description_lines=[
                    row["description"],
                    f"Unlocked: {row['unlocked']} | Trained: {row['trained']} | Validated best: {row.get('official_best_model', False)}",
                    f"Skill: {row.get('learning_skill', 'NA')} | Eval success: {row.get('validation_success_rate', 0.0):.1%} | Episodes: {row['last_total_episodes']}",
                    f"Last reason: {row.get('last_promotion_reason', '')[:150]}",
                ],
            )
        )

    return menu.run(
        title="Train Menu",
        subtitle=last_trained_label,
        buttons=buttons,
        footer_lines=["Click an unlocked stage. Next stages unlock only after a validated best model.", "ESC: back to main menu"],
        allow_escape_back=True,
    )


def show_train_mode_menu(stage_row: dict) -> str | None:
    stage_name = stage_row["name"]
    latest_available = bool(stage_row.get("latest_checkpoint_path"))
    interrupted_available = bool(stage_row.get("interrupted_checkpoint_path"))
    menu = MenuScreen(caption="RL Shooter - Training Mode")
    buttons = [
        MenuButton("Train headless / no visual", "start_headless", ["Fast default MaskablePPO training without pygame rendering."]),
        MenuButton("Train with visual rendering", "start_visual", ["Slower, useful for debugging behavior. F3 toggles raycasts; F5 saves."]),
        MenuButton("Continue latest checkpoint headless", "resume_latest_headless", ["Resume same-stage MaskablePPO state from latestTrainingState*.pt."], enabled=latest_available),
        MenuButton("Continue latest checkpoint visually", "resume_latest_visual", ["Resume latest same-stage MaskablePPO state with rendering enabled."], enabled=latest_available),
        MenuButton("Continue interrupted checkpoint headless", "resume_interrupted_headless", ["Resume from interruptedTrainingState*.pt after Ctrl+C or window stop."], enabled=interrupted_available),
        MenuButton("Continue interrupted checkpoint visually", "resume_interrupted_visual", ["Resume interrupted checkpoint with rendering enabled."], enabled=interrupted_available),
    ]
    return menu.run(
        title=stage_name,
        subtitle="Choose training mode",
        buttons=buttons,
        footer_lines=["Headless is recommended for real training. Visual mode is for inspection/debugging.", "ESC: back to Train menu"],
        allow_escape_back=True,
    )


def show_previous_stage_source_menu(
    current_stage_name: str,
    previous_stage_name: str,
    source_rows: list[dict],
) -> str | None:
    menu = MenuScreen(caption="RL Shooter - Previous Stage Source")

    buttons: list[MenuButton] = []
    for row in source_rows:
        status_text = "Available" if row["enabled"] else "Not available"
        buttons.append(
            MenuButton(
                label=row["label"],
                tag=row["source"],
                enabled=bool(row["enabled"]),
                description_lines=[
                    row["description"],
                    f"Status: {status_text}",
                ],
            )
        )

    return menu.run(
        title=f"{current_stage_name} - previous-stage source",
        subtitle=f"Initialize from {previous_stage_name}",
        buttons=buttons,
        footer_lines=[
            "Validated best is the official curriculum path. Latest/interrupted are manual experiment options only.",
            "ESC: back to Train menu",
        ],
        allow_escape_back=True,
    )


def show_test_menu(model_rows: list[dict]) -> str | None:
    menu = MenuScreen(caption="RL Shooter - Test")

    if not model_rows:
        buttons = [MenuButton("No trained bots found", "none", ["Train and validate a stage first."], enabled=False)]
    else:
        buttons = [
            MenuButton(
                label=row["label"],
                tag=row["stage_id"],
                description_lines=[
                    row["description"],
                    f"Official validated model: {row.get('official', False)} | Passed validation: {row.get('passed_validation', False)}",
                ],
                enabled=True,
            )
            for row in model_rows
        ]

    return menu.run(
        title="Watch / Test Menu",
        subtitle="Choose which saved RL bot to fight",
        buttons=buttons,
        footer_lines=["ESC: back to main menu"],
        allow_escape_back=True,
    )
