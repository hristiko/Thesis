from __future__ import annotations

import pygame

from config.stages import get_stage
from game.session import run_human_vs_model_bot, run_human_vs_stock_bot
from tools.sanity_check import run_stage0_sanity_check
from training.stage_manager import (
    ensure_stage_stats_file,
    get_last_trained_stage_label,
    get_previous_stage_name,
    get_previous_stage_source_rows,
    get_stage_display_rows,
    get_testable_model_rows,
)
from training.train import run_training_for_stage
from ui.menu import (
    show_main_menu,
    show_previous_stage_source_menu,
    show_test_menu,
    show_train_menu,
    show_train_mode_menu,
)


def _row_for_stage(stage_rows: list[dict], stage_id: str) -> dict | None:
    for row in stage_rows:
        if row["stage_id"] == stage_id:
            return row
    return None


def _run_selected_training(stage_choice: str, stage_row: dict, mode_choice: str) -> None:
    if stage_choice == "stage0":
        run_stage0_sanity_check(verbose=True)
        return

    stage = get_stage(stage_choice)
    visible = mode_choice.endswith("visual")

    if mode_choice.startswith("resume_latest"):
        checkpoint = stage_row.get("latest_checkpoint_path")
        if checkpoint:
            run_training_for_stage(stage_choice, visible=visible, resume_checkpoint=checkpoint)
        return

    if mode_choice.startswith("resume_interrupted"):
        checkpoint = stage_row.get("interrupted_checkpoint_path")
        if checkpoint:
            run_training_for_stage(stage_choice, visible=visible, resume_checkpoint=checkpoint)
        return

    prev_source = "best"
    if stage.required_previous_stage is not None:
        previous_stage_name = get_previous_stage_name(stage_choice)
        source_choice = show_previous_stage_source_menu(
            current_stage_name=stage.name,
            previous_stage_name=previous_stage_name or "previous stage",
            source_rows=get_previous_stage_source_rows(stage_choice),
        )
        if source_choice is None or source_choice == "quit":
            return
        prev_source = source_choice

    run_training_for_stage(
        stage_choice,
        visible=visible,
        resume_same_stage=False,
        prev_source=prev_source,
    )


def main() -> None:
    ensure_stage_stats_file()

    running = True
    while running:
        choice = show_main_menu()

        if choice == "play_stock":
            run_human_vs_stock_bot()

        elif choice == "sanity":
            run_stage0_sanity_check(verbose=True)

        elif choice == "train":
            while True:
                stage_rows = get_stage_display_rows()
                stage_choice = show_train_menu(stage_rows, get_last_trained_stage_label())

                if stage_choice is None or stage_choice == "quit":
                    break

                stage_row = _row_for_stage(stage_rows, stage_choice)
                if stage_row is None:
                    continue

                if stage_choice == "stage0":
                    run_stage0_sanity_check(verbose=True)
                    continue

                mode_choice = show_train_mode_menu(stage_row)
                if mode_choice is None or mode_choice == "quit":
                    continue

                _run_selected_training(stage_choice, stage_row, mode_choice)

        elif choice == "test":
            stage_choice = show_test_menu(get_testable_model_rows())
            if stage_choice == "quit":
                running = False
            elif stage_choice and stage_choice != "none":
                run_human_vs_model_bot(stage_choice)

        elif choice == "quit":
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
