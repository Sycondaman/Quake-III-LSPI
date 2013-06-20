#pragma once

// Includes both information needed to calculate the basis function and the reward
typedef struct lspi_action_basis_s
{
	// For calculating reward
	int kill_diff;
	int death_diff;
	int health_diff; // Diff between current health and last frame
	int armor_diff; // Diff between current armor and last frame
	int hit_count_diff;

	// Used to calculate diffs
	int kills;
	int deaths;

	// Stats
	int stat_health;
	int stat_armor;
	int stat_max_health;

	// Powerups, each represents the seconds remaining
	int pw_quad;
	int pw_battlesuit;
	int pw_haste;
	int pw_invis;
	int pw_regen;
	int pw_flight;
	int pw_scout;
	int pw_guard;
	int pw_doubler;
	int pw_ammoregen;
	int pw_invulnerability;

	// Ammo
	int wp_gauntlet;
	int wp_machinegun;
	int wp_shotgun;
	int wp_grenade_launcher;
	int wp_rocket_launcher;
	int wp_lightning;
	int wp_railgun;
	int wp_plasmagun;
	int wp_bfg;
	int wp_grappling_hook;

	// Enemy Information
	int enemy; // From the basis function -1 if there is no enemy, 1 otherwise
	float enemy_line_dist; // Straight line distance between bot and last known enemy pos
	float enemyposition_time; // From bot state, last time the enemy was seen
	int enemy_is_invisible; // Return value from EntityIsInvisible
	int enemy_is_shooting; // Return value from EntityIsShooting
	int enemy_weapon; // From EntityInfo

	// Goal information
	int goal_flags; // From bot_goal_t
	int item_type; // Gathered from iteminfo in bot_goal_t

	// Exit information
	int last_enemy_area_exits; // Number of paths into/out of last known enemy area
	int goal_area_exits;
	int current_area_exits;

	// Area numbers.. form functions by comparing which areas overlap
	int current_area_num;
	int goal_area_num;
	int enemy_area_num;

	// Misc
	int tfl; // Travel flags from bot_state
	int last_hit_count; // Number of times bot hit an enemy in the last frame
} lspi_action_basis_t;

typedef struct sample_t
{
	lspi_action_basis_t *state;
	lspi_action_basis_t *final_state;
	int action;
} sample;