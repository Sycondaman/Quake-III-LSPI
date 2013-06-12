/**
 * The base class for AI Agents. Represents a random agent.
 */

#pragma once

#include "constants.h"

class Agent
{
	public:
		/**
		 * Selects an action based on the given angle, x, and angular velocity, v.
		 * By default, selects an action at random.
		 */
		int getAction(double, double)
		{
			int choice = rand() % 3;
			switch(choice)
			{
				case 0:
					return NF_OPT;
				case 1:
					return LF_OPT;
				case 2:
					return RF_OPT;
			}
			
			return NF_OPT;
		}

		/**
		 * Notifies the agent of the reward for its actions.
		 * By default, does nothing.
		 */
		void notify(double)
		{
		}
};