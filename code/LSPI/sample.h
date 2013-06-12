#pragma once

#include "..\game\be_ai_lspi.h"

struct sample
{
	lspi_action_basis_t *state;
	lspi_action_basis_t *final_state;
	int action;
};