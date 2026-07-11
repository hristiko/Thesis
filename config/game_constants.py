from __future__ import annotations

# CS2D-compatible spatial scale.
TILE_SIZE_PX = 32
WALL_THICKNESS_PX = 32

# Player body must fit inside one 32x32 tile.
PLAYER_RADIUS_PX = 16.0
PLAYER_DIAMETER_PX = PLAYER_RADIUS_PX * 2.0

# Shared movement and Glock parameters. 160 px/s is exactly five 32px tiles/s.
PLAYER_MOVE_SPEED_PX_PER_SEC = 85
PLAYER_MAX_HEALTH = 100.0

GLOCK_DAMAGE_HP = 21.0
GLOCK_PROJECTILE_SPEED_PX_PER_SEC = 1500.0
GLOCK_FIRE_RATE = 2.0
GLOCK_RELOAD_TIME = 1.2
GLOCK_MAX_AMMO = 20

# Stage 3 boundary comfort is measured as clearance from the player's outer circle,
# not from the player center. 48 px clearance means a center about 64 px from the wall.
STAGE3_WALL_CLEARANCE_PX = 48.0
