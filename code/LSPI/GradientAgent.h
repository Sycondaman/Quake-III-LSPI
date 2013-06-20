#pragma once

/**
 * Provides an implementation of OLPOMDP (Baxter & Bartlett, 2000)
 *
 * Two template implementations are provided and only these two implementations should be used:
 * *thrust::host_vector
 * *thrust::device_vector
 */

#include "stdafx.h"
#include "sample.h"
#include <vector>
#include "blas.h"
#include "Matrix.h"
#include <thrust\host_vector.h>
#include "..\game\be_ai_lspi.h"
#include <Windows.h>
#include <stdlib.h>
#include <cmath>

#define SIGMA_2 1
#define E_CONST 2.71828f

#define PRINT(X) do															 \
	{																		 \
		printf("\n");														 \
		for(int z = 0; z < X.size(); z++) { printf("%.6f\n", (float)X[z]); } \
		printf("\n");														 \
    } while(0)

using namespace thrust;
using namespace blas;

template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
{
__host__ __device__ T operator()(const T &x) const
{
    return x < T(0) ? -x : x;
}
};

template <typename vector_type>
class GradientAgent
{
	public:
		vector_type w;

		GradientAgent(thrust::host_vector<float> policy, float stepsize, float bias) : w(policy), 
			gamma(stepsize), beta(bias), eligibility(BASIS_SIZE*NUM_ACTIONS) 
		{
			thrust::fill(eligibility.begin(), eligibility.end(), 0.0f);
		}

		/**
		 * Updates the policy online given the sample s. This algorithm assumes that s was dervied
		 * from the initial state and an action specified by getAction(state).
		 */
		void update(sample s)
		{
			vector_type basis = basis_function(s.state, s.action);
			vector_type basis_prime(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(basis_prime.begin(), basis_prime.end(), 0.0f);
			float reward = calculateReward(s);
			
			if(s.final_state->enemy == -1)
			{
				vector_type basis_ltg = basis_function(s.final_state, LSPI_LTG);
				vector_type basis_nbg = basis_function(s.final_state, LSPI_NBG);

				// Calculate dem probabilities
				float nbg = getPartialProbability(s.final_state, LSPI_NBG);
				float ltg = getPartialProbability(s.final_state, LSPI_LTG);
				float total = nbg + ltg;
				nbg = nbg/total;
				ltg = ltg/total;

				// Now we compute basis_prime = nbg*basis_nbg + ltg*basis_ltg
				axpy(basis_nbg, basis_prime, nbg);
				axpy(basis_ltg, basis_prime, ltg);
			}
			else
			{
				vector_type basis_chase = basis_function(s.final_state, LSPI_CHASE);
				vector_type basis_bnbg = basis_function(s.final_state, LSPI_BATTLE_NBG);
				vector_type basis_fight = basis_function(s.final_state, LSPI_FIGHT);
				vector_type basis_retreat = basis_function(s.final_state, LSPI_RETREAT);

				// Calculate dem probabilities
				float chase = getPartialProbability(s.final_state, LSPI_CHASE);
				float bnbg = getPartialProbability(s.final_state, LSPI_BATTLE_NBG);
				float fight = getPartialProbability(s.final_state, LSPI_FIGHT);
				float retreat = getPartialProbability(s.final_state, LSPI_RETREAT);
				float total = chase + bnbg + fight + retreat;
				chase = chase/total;
				bnbg = bnbg/total;
				fight = fight/total;
				retreat = retreat/total;

				// Now compute basis_prime = chase*basis_chase + bnbg*basis_bnbg + fight*basis_fight
				// + retreat*basis_retreat
				axpy(basis_chase, basis_prime, chase);
				axpy(basis_bnbg, basis_prime, bnbg);
				axpy(basis_fight, basis_prime, fight);
				axpy(basis_retreat, basis_prime, retreat);
			}

			// The update step for e: e <- beta*e + basis - basis_prime
			scal(eligibility, beta);
			axpy(basis, eligibility);
			axpy(basis_prime, eligibility, -1.0);

			// The update step for w: w <- w + gamma*reward*e
			vector_type step(eligibility);
			axpy(step, w, gamma*reward);
		}

		/**
		 * After creation, the LSPI Agent's policy is used to generate a functional value at a given angle and velocity. This functional value defines the action
		 * the agent intends to take.
		 */
		int getAction(lspi_action_basis_t *state)
		{
			float nbg, ltg, chase, bnbg, fight, retreat, total;
			vector_type params;

			if(state->enemy == -1)
			{
				nbg = getPartialProbability(state, LSPI_NBG);
				ltg = getPartialProbability(state, LSPI_LTG);

				// Divide the basis results by the total to determine the probability distribution
				total = nbg + ltg;
				nbg = nbg/total;
				ltg = ltg/total;

				// Now select one of these options
				float select = (float)rand()/RAND_MAX;
				if(select < nbg)
				{
					return LSPI_NBG;
				}
				else
				{
					return LSPI_LTG;
				}
			}
			else
			{
				chase = getPartialProbability(state, LSPI_CHASE);
				bnbg = getPartialProbability(state, LSPI_BATTLE_NBG);
				fight = getPartialProbability(state, LSPI_FIGHT);
				retreat = getPartialProbability(state, LSPI_RETREAT);

				// Divide the basis results by the total to determine the probability
				total = chase + bnbg + fight + retreat;
				chase = chase/total;
				bnbg = bnbg/total;
				fight = fight/total;
				retreat = retreat/total;

				// Now select one of these options
				float select = (float)rand()/RAND_MAX;
				if(select < chase)
				{
					return LSPI_CHASE;
				}

				select -= chase;
				if(select < bnbg)
				{
					return LSPI_BATTLE_NBG;
				}

				select -= bnbg;
				if(select < fight)
				{
					return LSPI_FIGHT;
				}

				return LSPI_RETREAT;
			}
		}

	private:
		float gamma, beta;
		vector_type eligibility;

		/**
		 * Calculates e^J_theta(s, a), the partial probability of the action for use in the boltzmann distribution.
		 */
		float getPartialProbability(lspi_action_basis_t *state, int action)
		{
			float val, temp;
			vector_type params = basis_function(state, action);
			dot(params, w, val);

			float rval = pow(E_CONST, (float)(val/1e6));
			return rval;
		}

		/** 
		 * Calculates the reward earned for a given (s,a,s') tuple.
		 */
		float calculateReward(sample s)
		{
			float r_health = 0.01 * (float)(s.final_state->health_diff);
			float r_hit = 0.5 * (float)(s.final_state->hit_count_diff);
			float r_armor = 0.005 * (float)(s.final_state->armor_diff);
			float r_kill = 2 * (float)(s.final_state->kill_diff);
			float r_death = -2 * (float)(s.final_state->death_diff);

			return r_health + r_hit + r_armor + r_kill + r_death - 0.001;
		}
	
		/**
		 * Returns the policy function weights for the given angle, velocity, and action.
		 * These weights can be used to compute the estimated fitness of the given action.
		 */
		// LARGE BASIS FUNCIONT (6*43)
//		vector_type basis_function(lspi_action_basis_t *state, int action)
//		{
//			vector_type phi(BASIS_SIZE*NUM_ACTIONS);
//			thrust::fill(phi.begin(), phi.end(), 0.0f);
//			
//#if defined(VERBOSE_HIGH)
//			PRINT(phi);
//#endif
//
//			// TODO: Move this into a transform/cuda kernel
//			// Now populate the basis function for this state action pair
//			// Note that each entry except for the first is a gaussian.
//			int i = BASIS_SIZE * (action-1);
//			phi[i] = 1.0f;
//			
//			// Health
//			float over_health = state->stat_health - state->stat_max_health;
//			if(over_health >= 0)
//			{
//				phi[i+1] = over_health;
//				phi[i+2] = 1; // Max health
//			}
//			else
//			{
//				float percent_health = (float)state->stat_health/state->stat_max_health;
//				if(percent_health < 0.20)
//				{
//					phi[i+3] = 1; // Critical health
//				}
//				else if(percent_health < 0.50)
//				{
//					phi[i+4] = 1;
//				}
//				else
//				{
//					phi[i+5] = 1;
//				}
//			}
//
//			// Armor
//			float over_armor = state->stat_armor - 100;
//			if(over_armor >= 0)
//			{
//				phi[i+6] = over_armor;
//				phi[i+7] = 1.0;
//			}
//			else
//			{
//				float percent_armor = (float)state->stat_armor/100.0f;
//				phi[i+8] = percent_armor;
//			}
//
//			// Powerups
//			phi[i+9] = state->pw_quad;
//			phi[i+10] = state->pw_battlesuit;
//			phi[i+11] = state->pw_haste;
//			phi[i+12] = state->pw_invis;
//			phi[i+13] = state->pw_regen;
//			phi[i+14] = state->pw_flight;
//			phi[i+15] = state->pw_scout;
//			phi[i+16] = state->pw_guard;
//			phi[i+17] = state->pw_doubler;
//			phi[i+18] = state->pw_ammoregen;
//			phi[i+19] = state->pw_invulnerability;
//
//			// enemy
//			phi[i+20] = state->enemy;
//			if(state->enemy_line_dist < 500)
//			{
//				phi[i+21] = 1;
//			}
//			else if(state->enemy_line_dist > 1500)
//			{
//				phi[i+22] = 1;
//			}
//			else
//			{
//				phi[i+23] = 1;
//			}
//			phi[i+24] = state->enemy_is_invisible;
//			phi[i+25] = state->enemy_is_shooting;
//
//			phi[i+26] = state->enemy_area_num;
//			phi[i+27] = state->current_area_num;
//			phi[i+28] = state->goal_area_num;
//
//			// Ammo information
//			phi[i+29] = state->wp_gauntlet;
//			phi[i+30] = state->wp_machinegun;
//			phi[i+31] = state->wp_shotgun;
//			phi[i+32] = state->wp_grenade_launcher;
//			phi[i+33] = state->wp_rocket_launcher;
//			phi[i+34] = state->wp_lightning;
//			phi[i+35] = state->wp_railgun;
//			phi[i+36] = state->wp_plasmagun;
//			phi[i+37] = state->wp_bfg;
//			phi[i+38] = state->wp_grappling_hook;
//
//			// Shared location information
//			phi[i+39] = state->enemy_area_num == state->goal_area_num ? 1 : 0;
//			phi[i+40] = state->enemy_area_num == state->current_area_num ? 1 : 0;
//			phi[i+41] = state->goal_area_num == state->current_area_num ? 1 : 0;
//			phi[i+42] = phi[i+41] == phi[i+40] ? 1 : 0;
//
//#if defined(VERBOSE_HIGH)
//			PRINT(phi);
//#endif
//
//			return phi;
//		}


// MEDIUM BASIS FUNCTION (6*29)
//		vector_type basis_function(lspi_action_basis_t *state, int action)
//		{
//			vector_type phi(BASIS_SIZE*NUM_ACTIONS);
//			thrust::fill(phi.begin(), phi.end(), 0.0f);
//			
//#if defined(VERBOSE_HIGH)
//			PRINT(phi);
//#endif
//
//			// TODO: Move this into a transform/cuda kernel
//			// Now populate the basis function for this state action pair
//			// Note that each entry except for the first is a gaussian.
//			int i = BASIS_SIZE * (action-1);
//			phi[i] = 1.0f;
//			
//			// Health
//			float over_health = state->stat_health - state->stat_max_health;
//			if(over_health >= 0)
//			{
//				phi[i+1] = over_health;
//				phi[i+2] = 1; // Max health
//			}
//			else
//			{
//				float percent_health = (float)state->stat_health/state->stat_max_health;
//				if(percent_health < 0.20)
//				{
//					phi[i+3] = 1; // Critical health
//				}
//				else if(percent_health < 0.50)
//				{
//					phi[i+4] = 1;
//				}
//				else
//				{
//					phi[i+5] = 1;
//				}
//			}
//
//			// Armor
//			float over_armor = state->stat_armor - 100;
//			if(over_armor >= 0)
//			{
//				phi[i+6] = over_armor;
//				phi[i+7] = 1.0;
//			}
//			else
//			{
//				float percent_armor = (float)state->stat_armor/100.0f;
//				phi[i+8] = percent_armor;
//			}
//
//			// Powerups
//			phi[i+9] = state->pw_quad;
//			phi[i+10] = state->pw_battlesuit;
//			phi[i+11] = state->pw_haste;
//			phi[i+12] = state->pw_invis;
//			phi[i+13] = state->pw_regen;
//			phi[i+14] = state->pw_flight;
//			phi[i+15] = state->pw_scout;
//			phi[i+16] = state->pw_guard;
//			phi[i+17] = state->pw_doubler;
//			phi[i+18] = state->pw_ammoregen;
//			phi[i+19] = state->pw_invulnerability;
//
//			// enemy
//			phi[i+20] = state->enemy;
//			if(state->enemy_line_dist < 500)
//			{
//				phi[i+21] = 1;
//			}
//			else if(state->enemy_line_dist > 1500)
//			{
//				phi[i+22] = 1;
//			}
//			else
//			{
//				phi[i+23] = 1;
//			}
//			phi[i+24] = state->enemy_is_invisible;
//			phi[i+25] = state->enemy_is_shooting;
//
//			phi[i+26] = state->enemy_area_num;
//			phi[i+27] = state->current_area_num;
//			phi[i+28] = state->goal_area_num;
//
//#if defined(VERBOSE_HIGH)
//			PRINT(phi);
//#endif
//
//			return phi;
//		}


/// SMALL BASIS FUNCTION (6*5)
		vector_type basis_function(lspi_action_basis_t *state, int action)
		{
			vector_type phi(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(phi.begin(), phi.end(), 0.0f);
			
#if defined(VERBOSE_HIGH)
			PRINT(phi);
#endif

			// TODO: Move this into a transform/cuda kernel
			// Now populate the basis function for this state action pair
			// Note that each entry except for the first is a gaussian.
			int i = BASIS_SIZE * (action-1);
			phi[i] = 1.0f;
			
			phi[i+1] = state->stat_health;
			phi[i+2] = state->stat_armor;
			phi[i+3] = state->enemy;
			phi[i+4] = state->enemy_line_dist;

#if defined(VERBOSE_HIGH)
			PRINT(phi);
#endif

			return phi;
		}
};

