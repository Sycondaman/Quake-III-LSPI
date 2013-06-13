#pragma once

#include "q_shared.h"
#include "ai_main.h"

//#define CPU
//#define GRADIENT
//#define EXPLORE
#define EXP_RATE 0.25f

#define LSPI_NBG 1
#define LSPI_LTG 2
#define LSPI_CHASE 3
#define LSPI_BATTLE_NBG 4
#define LSPI_FIGHT 5
#define LSPI_RETREAT 6

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

// Copies the relevant data from the the bot_state to the basis
//void LspiBot_PopulateBasisFromState(lspi_action_basis_t *basis, bot_state_t *bs);
#ifdef __cplusplus
extern "C" {
#endif

// Perform the initial load of the agent for the given client
void LspiBot_Init(int client);

void LspiBot_Shutdown(int client);

// Gets the recommended action for the specified client + state
int LspiBot_GetAction(int client, lspi_action_basis_t *basis);

// Does nothing for an LSPI Bot, performs the OLPOMDP update for a gradient bot
void LspiBot_GradUpdate(int client, lspi_action_basis_t *prev, lspi_action_basis_t *cur, int action);

// Updates the LspiBot policy
void LspiBot_Update(int client, const char *fname);
#ifdef __cplusplus
}
#endif