#include "stdafx.h"
#include "..\game\be_ai_lspi.h"
#include <stdlib.h>
#include <fstream>
#include <string>
#include "LspiAgent.h"
#include "GradientAgent.h"
#include <thrust/generate.h>
#include <sys/stat.h>
#include <iomanip>
#include <windows.h>

#ifdef GRADIENT
	#ifdef CPU
		GradientAgent<host_vector<float>> *agents[MAX_CLIENTS];
	#else
		GradientAgent<device_vector<float>> *agents[MAX_CLIENTS];
	#endif
#else
	#ifdef CPU
		LspiAgent<host_vector<float>> *agents[MAX_CLIENTS];
	#else
		LspiAgent<device_vector<float>> *agents[MAX_CLIENTS];
	#endif
#endif

#ifdef ONLINE
struct threadarg
{
	int client;
	sample *samples;
	int size;
};

HANDLE updateMutex, threadHandle;
bool updateComplete = true;
__int64 frequency;
#endif

using namespace std;

/*
 * Loads the bot's policy from a file and spawns an LspiAgent using the policy.
 */
void LspiBot_Init(int client)
{
#ifdef ONLINE
	if(updateMutex == NULL)
	{
		updateMutex = CreateMutex(NULL, FALSE, NULL);
	}

	LARGE_INTEGER li;
	if(QueryPerformanceFrequency(&li))
	{
		frequency = li.QuadPart;
	}
#endif

	host_vector<float> policy(NUM_ACTIONS*BASIS_SIZE);
	string value;
#ifdef GRADIENT
	char *fname = "grad.pol";
#else
	char *fname = "lspi.pol";
#endif
	struct stat buf;
	ifstream infile;
	ofstream outfile;

	if(stat(fname, &buf) == -1)
	{
		outfile.open(fname);
		for(int i = 0; i < policy.size(); i++)
		{
			policy[i] = ((float)rand()/RAND_MAX);
			if(i + 1 == policy.size())
			{
				outfile << fixed << setprecision(8) << policy[i] << endl;
			}
			else
			{
				outfile << fixed << setprecision(8) << policy[i] << ",";
			}
		}
		outfile.close();
	}
	else
	{
		infile.open(fname);
		for(int i = 0; i < NUM_ACTIONS*BASIS_SIZE; i++)
		{
			getline(infile, value, ',');
			policy[i] = (float)atof(value.c_str());
		}
		infile.close();
	}
#ifndef CPU
	cublasStatus_t stat = cublasCreate(&blas::handle);
#endif

#ifdef EXPLORE
	bool explore = true;
#else
	bool explore = false;
#endif

#ifdef GRADIENT
	#ifdef CPU
		agents[client] = new GradientAgent<host_vector<float>>(policy, 0.01, 0.80);
	#else
		agents[client] = new GradientAgent<device_vector<float>>(policy, 0.01, 0.80);
	#endif
#else
	#ifdef CPU
		agents[client] = new LspiAgent<host_vector<float>>(policy, 0.95, explore, EXP_RATE);
	#else
		agents[client] = new LspiAgent<device_vector<float>>(policy, 0.95, explore, EXP_RATE);
	#endif
#endif
}

void LspiBot_Shutdown(int client)
{
#ifdef GRADIENT
	ofstream outfile;
	char *fname = "grad.pol";
	float temp;

	outfile.open(fname);
	for(int i = 0; i < agents[client]->w.size(); i++)
	{
		temp = agents[client]->w[i];
		if(i + 1 == agents[client]->w.size())
		{
			outfile << fixed << setprecision(8) << agents[client]->w[i] << endl;
		}
		else
		{
			outfile << fixed << setprecision(8) << agents[client]->w[i] << ",";
		}
	}
	outfile.close();
#endif
#ifdef ONLINE
	SuspendThread(threadHandle);
	CloseHandle(threadHandle);
	CloseHandle(updateMutex);
#endif

	delete agents[client];
}

int LspiBot_GetAction(int client, lspi_action_basis_t *basis) {
	return agents[client]->getAction(basis);
}

void LspiBot_GradUpdate(int client, lspi_action_basis_t *prev, lspi_action_basis_t *cur, int action)
{
#ifdef GRADIENT
	sample s;
	s.state = cur;
	s.action = action;
	s.final_state = prev;

	agents[client]->update(s);
#endif
}

#ifdef ONLINE
	DWORD WINAPI UpdateThread( LPVOID lpParam )
	{
		LARGE_INTEGER before, after;
	
		QueryPerformanceCounter(&before);
		threadarg *args = (threadarg*)lpParam;
		host_vector<sample> samples(args->size);
		for(int i = 0; i < args->size; i++)
		{
			sample s;
			s.state = args->samples[i].state;
			s.final_state = args->samples[i].final_state;
			s.action = args->samples[i].action;
			samples[i] = s;
		}

		device_vector<sample> dev_samples = samples;
		host_vector<float> policy = agents[args->client]->updatePolicy(dev_samples);

		// Write to file
		ofstream outfile("lspi.pol");
		for(int i = 0; i < policy.size(); i++)
		{
			if(i + 1 == policy.size())
			{
				outfile << fixed << setprecision(8) << policy[i] << endl;
			}
			else
			{
				outfile << fixed << setprecision(8) << policy[i] << ",";
			}
		}
		outfile.close();

		QueryPerformanceCounter(&after);
		double policy_update_time = (double)(after.QuadPart - before.QuadPart)/frequency;

		ofstream perffile;
		char *fname = "perf_online.dat";

		outfile.open(fname, ofstream::app);
		outfile << "Policy Update (Size, Time): " << args->size << ", " << fixed << setprecision(8) << 1000.0*policy_update_time << endl;
		outfile.close();
		
		free(args);

		WaitForSingleObject(updateMutex, INFINITE);
		updateComplete = true;
		ReleaseMutex(updateMutex);

		return 0;
	}

	int LspiBot_Update(int client, sample *samples, int size)
	{
		WaitForSingleObject(updateMutex, INFINITE);

		if(updateComplete)
		{
			if(threadHandle != NULL)
			{
				CloseHandle(threadHandle);
			}

			threadarg *args = (threadarg *)malloc(sizeof(threadarg));
			args->client = client;
			args->samples = samples;
			args->size = size;

			threadHandle = CreateThread(NULL, 0, UpdateThread, args, 0, NULL); 
			if(threadHandle == NULL)
			{
				int test_val; // We can't print anything out from here... so let's just make something we can put a stop point on.
			}
			updateComplete = false;

			ReleaseMutex(updateMutex);
			return 1;
		}

		ReleaseMutex(updateMutex);
		return 0;
	}
#else
	void LspiBot_Update(int client, const char *fname)
	{
	#ifndef GRADIENT
		// Load the samples into a vector and update LSPI agent's policy
		host_vector<sample> samples;
		string value;
		ifstream file(fname);

		thrust::host_vector<sample>::iterator it = samples.end(); 
		while(file.good())
		{
			sample s;
			lspi_action_basis_t *state = (lspi_action_basis_t*)malloc(sizeof(lspi_action_basis_t));
			lspi_action_basis_t *fstate = (lspi_action_basis_t*)malloc(sizeof(lspi_action_basis_t));
			s.state = state;
			s.final_state = fstate;

			//// Action ////
			if(!getline(file, value, ','))
			{
				break;
			}
			s.action = atoi(value.c_str());
			////////////////

			/***** START READING STATE *****/

			//// For calculated reward ////
			getline(file, value, ',');
			state->kill_diff = atoi(value.c_str());

			getline(file, value, ',');
			state->death_diff = atoi(value.c_str());

			getline(file, value, ',');
			state->health_diff = atoi(value.c_str());

			getline(file, value, ',');
			state->armor_diff = atoi(value.c_str());

			getline(file, value, ',');
			state->hit_count_diff = atoi(value.c_str());
			///////////////////////////////

			//// Stats ////
			getline(file, value, ',');
			state->stat_health = atoi(value.c_str());

			getline(file, value, ',');
			state->stat_armor = atoi(value.c_str());

			getline(file, value, ',');
			state->stat_max_health = atoi(value.c_str());
			///////////////

			//// Powerups ////
			getline(file, value, ',');
			state->pw_quad = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_battlesuit = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_haste = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_invis = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_regen = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_flight = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_scout = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_guard = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_doubler = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_ammoregen = atoi(value.c_str());

			getline(file, value, ',');
			state->pw_invulnerability = atoi(value.c_str());
			//////////////////

			//// Ammo ////
			getline(file, value, ',');
			state->wp_gauntlet = atoi(value.c_str());

			getline(file, value, ',');
			state->wp_machinegun = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_shotgun = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_grenade_launcher = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_rocket_launcher = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_lightning = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_railgun = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_plasmagun = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_bfg = atoi(value.c_str());
		
			getline(file, value, ',');
			state->wp_grappling_hook = atoi(value.c_str());
			//////////////

			//// Enemy Info ////
			getline(file, value, ',');
			state->enemy = atoi(value.c_str());

			getline(file, value, ',');
			state->enemy_line_dist = (float)atof(value.c_str());

			getline(file, value, ',');
			state->enemyposition_time = (float)atof(value.c_str());

			getline(file, value, ',');
			state->enemy_is_invisible = atoi(value.c_str());

			getline(file, value, ',');
			state->enemy_is_shooting = atoi(value.c_str());

			getline(file, value, ',');
			state->enemy_weapon = atoi(value.c_str());
			////////////////////

			//// Goal Info////
			getline(file, value, ',');
			state->goal_flags = atoi(value.c_str());

			getline(file, value, ',');
			state->item_type = atoi(value.c_str());
			//////////////////

			//// Exit Information ////
			getline(file, value, ',');
			state->last_enemy_area_exits = atoi(value.c_str());

			getline(file, value, ',');
			state->goal_area_exits = atoi(value.c_str());

			getline(file, value, ',');
			state->current_area_exits = atoi(value.c_str());
			//////////////////////////
		
			//// Area Numbers ////
			getline(file, value, ',');
			state->current_area_num = atoi(value.c_str());

			getline(file, value, ',');
			state->goal_area_num = atoi(value.c_str());

			getline(file, value, ',');
			state->enemy_area_num = atoi(value.c_str());
			//////////////////////////

			//// Misc ////
			getline(file, value, ',');
			state->tfl = atoi(value.c_str());

			getline(file, value, ',');
			state->last_hit_count = atoi(value.c_str());
			//////////////
		
			/***** END READING STATE *****/

			/***** START READING FINAL STATE *****/

			//// For calculated reward ////
			getline(file, value, ',');
			fstate->kill_diff = atoi(value.c_str());

			getline(file, value, ',');
			fstate->death_diff = atoi(value.c_str());

			getline(file, value, ',');
			fstate->health_diff = atoi(value.c_str());

			getline(file, value, ',');
			fstate->armor_diff = atoi(value.c_str());

			getline(file, value, ',');
			state->hit_count_diff = atoi(value.c_str());
			///////////////////////////////

			//// Stats ////
			getline(file, value, ',');
			fstate->stat_health = atoi(value.c_str());

			getline(file, value, ',');
			fstate->stat_armor = atoi(value.c_str());

			getline(file, value, ',');
			fstate->stat_max_health = atoi(value.c_str());
			///////////////

			//// Powerups ////
			getline(file, value, ',');
			fstate->pw_quad = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_battlesuit = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_haste = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_invis = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_regen = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_flight = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_scout = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_guard = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_doubler = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_ammoregen = atoi(value.c_str());

			getline(file, value, ',');
			fstate->pw_invulnerability = atoi(value.c_str());
			//////////////////

			//// Ammo ////
			getline(file, value, ',');
			fstate->wp_gauntlet = atoi(value.c_str());

			getline(file, value, ',');
			fstate->wp_machinegun = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_shotgun = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_grenade_launcher = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_rocket_launcher = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_lightning = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_railgun = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_plasmagun = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_bfg = atoi(value.c_str());
		
			getline(file, value, ',');
			fstate->wp_grappling_hook = atoi(value.c_str());
			//////////////

			//// Enemy Info ////
			getline(file, value, ',');
			fstate->enemy = atoi(value.c_str());

			getline(file, value, ',');
			fstate->enemy_line_dist = (float)atof(value.c_str());

			getline(file, value, ',');
			fstate->enemyposition_time = (float)atof(value.c_str());

			getline(file, value, ',');
			fstate->enemy_is_invisible = atoi(value.c_str());

			getline(file, value, ',');
			fstate->enemy_is_shooting = atoi(value.c_str());

			getline(file, value, ',');
			fstate->enemy_weapon = atoi(value.c_str());
			////////////////////

			//// Goal Info////
			getline(file, value, ',');
			fstate->goal_flags = atoi(value.c_str());

			getline(file, value, ',');
			fstate->item_type = atoi(value.c_str());
			//////////////////

			//// Exit Information ////
			getline(file, value, ',');
			fstate->last_enemy_area_exits = atoi(value.c_str());

			getline(file, value, ',');
			fstate->goal_area_exits = atoi(value.c_str());

			getline(file, value, ',');
			fstate->current_area_exits = atoi(value.c_str());
			//////////////////////////
		
			//// Area Numbers ////
			getline(file, value, ',');
			fstate->current_area_num = atoi(value.c_str());

			getline(file, value, ',');
			fstate->goal_area_num = atoi(value.c_str());

			getline(file, value, ',');
			fstate->enemy_area_num = atoi(value.c_str());
			//////////////////////////

			//// Misc ////
			getline(file, value, ',');
			fstate->tfl = atoi(value.c_str());

			getline(file, value, '\n');
			fstate->last_hit_count = atoi(value.c_str());
			//////////////

			/***** END READING FINAL STATE *****/

			samples.insert(it, s);
			it = samples.end();
		}
		file.close();

	#ifdef CPU
		host_vector<float> policy = agents[client]->updatePolicy(samples);
	#else
		device_vector<sample> dev_samples = samples;
		host_vector<float> policy = agents[client]->updatePolicy(dev_samples);
	#endif

		// Write to file
		ofstream outfile("lspi.pol");
		for(int i = 0; i < policy.size(); i++)
		{
			if(i + 1 == policy.size())
			{
				outfile << fixed << setprecision(8) << policy[i] << endl;
			}
			else
			{
				outfile << fixed << setprecision(8) << policy[i] << ",";
			}
		}
		outfile.close();

		// Free space used by samples
		for(int i = 0; i < samples.size(); i++)
		{
			free(samples[i].final_state);
			free(samples[i].state);
		}
	#endif
	}
#endif