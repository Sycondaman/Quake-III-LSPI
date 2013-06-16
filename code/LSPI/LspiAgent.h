#pragma once

/**
 * Provides an implemenation of the Least Squared Policy Iteration algorithm for solving the inverted pendulum problem.
 *
 * Two template implementations are provided and only these two implementations should be used:
 * *thrust::host_vector
 * *thrust::device_vector
 */

#include "stdafx.h"
#include "sample.h"
#include "constants.h"
#include <vector>
#include "blas.h"
#include "Matrix.h"
#include <thrust\host_vector.h>
#include "..\game\be_ai_lspi.h"
#include <Windows.h>

#define NUM_ACTIONS 6
#define BASIS_SIZE 5
#define SIGMA_2 1

//#define VERBOSE_HIGH
//#define VERBOSE_MED
//#define VERBOSE_LOW

#if defined(VERBOSE_HIGH)
#	define VERBOSE_MED
#	define VERBOSE_LOW
#elif defined(VERBOSE_MED)
#	define VERBOSE_LOW
#endif

#define PRINT(X) do															 \
	{																		 \
		printf("\n");														 \
		for(int z = 0; z < X.size(); z++) { printf("%.6f\n", (float)X[z]); } \
		printf("\n");														 \
    } while(0)

using namespace thrust;
using namespace blas;

template <typename vector_type>
class LspiAgent
{
	public:
		LspiAgent(thrust::host_vector<float> policy, float disc, bool exp, float rate) : w(policy), discount(disc), explore(exp), exp_rate(rate) {}

		/**
		 * To create an LSPI Agent, a discount factor and a large number of sample data points are required. More sample should result in a better policy.
		 * The samples should come from data taken from an agent performing at random, or a previous iteration of an LSPI Agent.
		 *
		 * Each sample in the vector should be of the format [x, v, a, r, x', v', t]
		 * -x is the angle
		 * -v is the angular velocity
		 * -a is action selected
		 * -r is the reward received after executing the action
		 * -x' is the angle after executing the action
		 * -v' is the angular velocity after executing the action
		 * -t is 1 if the state after executing is terminal, 0 otherwise
		 */
#ifdef CPU
		thrust::host_vector<float> updatePolicy(thrust::host_vector<sample> samples)
#else
		thrust::host_vector<float> updatePolicy(thrust::device_vector<sample> samples)
#endif
		{
			// Loop until policy converges
			vector_type policy = lstdq(samples);
#if defined(VERBOSE_HIGH)
			PRINT(policy);
#endif
			vector_type temp(policy);
#if defined(VERBOSE_HIGH)
			PRINT(w);
#endif
			blas::axpy(w, temp, -1.0f);
#if defined(VERBOSE_HIGH)
			PRINT(temp);
#endif

			//TODO: Write a magnitude function dammit!
			float magnitude = 0.0f;
			for(int i = 0; i < temp.size(); i++)
			{
				magnitude += temp[i]*temp[i];
			}


			int k = 0;
			while(sqrt(magnitude) > epsilon_const && k < 10)
			{
				w = policy;
#if defined(VERBOSE_LOW)
				PRINT(w);
#endif
				policy = lstdq(samples);

				vector_type temp2(policy);
				blas::axpy(w, temp2, -1.0f);
				//TODO: Write a magnitude function dammit!
				magnitude = 0.0f;
				for(int i = 0; i < temp2.size(); i++)
				{
					magnitude += temp2[i]*temp2[i];
				}
				k++;
			}

#if defined(VERBOSE_LOW)
			PRINT(policy);
#endif

			w = policy;

			thrust::host_vector<float> rval(w);
			return rval;
		}

		/**
		 * After creation, the LSPI Agent's policy is used to generate a functional value at a given angle and velocity. This functional value defines the action
		 * the agent intends to take.
		 */
		int getAction(lspi_action_basis_t *state)
		{
			int action = -1;
			float max = -9e999;
			int i, j;

			if(state->enemy == -1)
			{
				if(explore)
				{
					if((float)rand()/RAND_MAX < exp_rate)
					{
						if((float)rand()/RAND_MAX < 0.50)
						{
							return 1;
						}
						else
						{
							return 2;
						}
					}
				}

				// Only #1 and #2 are valid options
				i = 1;
				j = 3;
			}
			else
			{
				if(explore)
				{
					if((float)rand()/RAND_MAX < exp_rate)
					{
						float select = (float)rand()/RAND_MAX;
						if(select < 0.25)
						{
							return 3;
						}
						else if(select < 0.50)
						{
							return 4;
						}
						else if(select < 0.75)
						{
							return 5;
						}
						return 6;
					}
				}
				// #3, 4, 5, 6 are valid
				i = 3;
				j = 7;
			}
			
			for(i; i < j; i++)
			{
				vector_type params = basis_function(state, i);
				float q; 
				dot(params, w, q);
				if(q > max)
				{
					action = i;
					max = q;
				}
			}

			return action;
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

	private:
		float discount, exp_rate;
		bool explore;
		vector_type w;
		
		// TODO: Test the speed penalty of copying samples to the GPU for gpu based implementation
		/**
		 * Given a set of samples, performs a single update step on the current agent's policy.
		 */
		vector_type lstdq(thrust::host_vector<sample> samples)
		{
			Matrix<vector_type> B(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(B.vector.begin(), B.vector.end(), 0.0f);

			// TODO: Put this in a function for both vector_types and write a custom CUDA kernel for the GPU implementation
			for(int i = 0; i < B.rows; i++)
			{
				for(int j = 0; j < B.rows; j++)
				{
					if(i == j)
						B.set(i, j, 1.0f);
				}
			}

			scal(B.vector, 0.1f); // TODO: Investigate the importance and effect of this number
			
#if defined(VERBOSE_HIGH)
			printf("\n");
			B.print();
#endif

			vector_type b(BASIS_SIZE*NUM_ACTIONS);
			thrust::fill(b.begin(), b.end(), 0.0f);

			for(unsigned int i = 0; i < samples.size(); i++)
			{
				float reward = calculateReward(samples[i]);

				// Get the basis functions
				vector_type phi = basis_function(samples[i].state, samples[i].action);
				int next_action = getAction(samples[i].final_state);
				vector_type phi_prime = basis_function(samples[i].final_state, next_action);

				// Break the calculation into smaller parts
				scal(phi_prime, discount);
				axpy(phi, phi_prime, -1.0f); // TODO: Consider optimizing this by creating a custom kernel
				scal(phi_prime, -1.0f); // This is because axpy does not allow us to do y = x - y, only y = y - x
				
#if defined(VERBOSE_HIGH)
				printf("\n");
				B.print();
				printf("\n");
				PRINT(phi);
				PRINT(phi_prime);
#endif

				// TODO: Try to eliminate extra memory allocation by reusing vectors
				vector_type temp(phi.size());
				vector_type temp2(phi.size());
				Matrix<vector_type> num(BASIS_SIZE*NUM_ACTIONS);
				thrust::fill(num.vector.begin(), num.vector.end(), 0.0f);

				gemv(B, phi, temp, false);
				gemv(B, phi_prime, temp2, true);
				ger(temp, temp2, num);
				
#if defined(VERBOSE_HIGH)
				PRINT(temp);
				PRINT(temp2);
				num.print();
#endif

				float denom;
				dot(phi, temp2, denom);
				denom += 1.0f;
				
#if defined(VERBOSE_HIGH)
				printf("\n%.3f\n", denom);
#endif

				scal(num.vector, 1.0f/denom);
				axpy(num.vector, B.vector, -1.0f);
				
#if defined(VERBOSE_HIGH)
				num.print();
				B.print();
#endif

				// Update values
				scal(phi, reward);
				axpy(phi, b);
				
#if defined(VERBOSE_HIGH)
				printf("\n%d\n", reward);

				PRINT(phi);
				PRINT(b);
#endif
			}

#if defined(VERBOSE_MED)
			B.print();
			PRINT(b);
#endif
			vector_type temp_b(b.size());
			gemv(B, b, temp_b, false);
			b = temp_b;
			
#if defined(VERBOSE_MED)
			PRINT(b);
#endif

			return b;
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

