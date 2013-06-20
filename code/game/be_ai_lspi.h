#pragma once

#include "q_shared.h"
#include "ai_main.h"
#include "..\LSPI\sample.h"

//#define CPU
//#define GRADIENT
#define ONLINE
#define EXPLORE
#define EXP_RATE 0.25f

#define LSPI_NBG 1
#define LSPI_LTG 2
#define LSPI_CHASE 3
#define LSPI_BATTLE_NBG 4
#define LSPI_FIGHT 5
#define LSPI_RETREAT 6



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
#ifdef ONLINE
int LspiBot_Update(int client, sample *samples, int size);
#else
void LspiBot_Update(int client, const char *fname);
#endif
#ifdef __cplusplus
}
#endif