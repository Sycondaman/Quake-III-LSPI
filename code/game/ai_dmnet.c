/*
===========================================================================
Copyright (C) 1999-2005 Id Software, Inc.

This file is part of Quake III Arena source code.

Quake III Arena source code is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the License,
or (at your option) any later version.

Quake III Arena source code is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
===========================================================================
*/
//

/*****************************************************************************
 * name:		ai_dmnet.c
 *
 * desc:		Quake3 bot AI
 *
 * $Archive: /MissionPack/code/game/ai_dmnet.c $
 *
 *****************************************************************************/

#include "g_local.h"
#include "botlib.h"
#include "be_aas.h"
#include "be_ea.h"
#include "be_ai_char.h"
#include "be_ai_chat.h"
#include "be_ai_gen.h"
#include "be_ai_goal.h"
#include "be_ai_move.h"
#include "be_ai_weap.h"
//
#include "ai_main.h"
#include "ai_dmq3.h"
#include "ai_chat.h"
#include "ai_cmd.h"
#include "ai_dmnet.h"
#include "ai_team.h"
//data file headers
#include "chars.h"			//characteristics
#include "inv.h"			//indexes into the inventory
#include "syn.h"			//synonyms
#include "match.h"			//string matching types and vars

// For LSPI and Gradient bot
#include "be_ai_lspi.h"

// for the voice chats
#include "../../ui/menudef.h"

// For saving basis info
#include <stdio.h>

// For perf metrics
#include <Windows.h>

//goal flag, see be_ai_goal.h for the other GFL_*
#define GFL_AIR			128

int numnodeswitches;
char nodeswitch[MAX_NODESWITCHES+1][144];

// Used to save sample data
lspi_action_basis_t *basis[MAX_CLIENTS];
lspi_action_basis_t *last_basis[MAX_CLIENTS]; 
int last_action[MAX_CLIENTS]; 

// Used to determine when to check LSPI/Gradient bot actions
int action_chosen[MAX_CLIENTS]; // Used to determine if an agent has chosen an action

// File pointers for saving data
FILE *sample_file[MAX_CLIENTS];
FILE *reward_file[MAX_CLIENTS];

// For tracking reward over time
float total_reward[MAX_CLIENTS];

// Perf metrics
double policy_update_time[MAX_CLIENTS];
long total_updates[MAX_CLIENTS];
double action_decision_time[MAX_CLIENTS];
long total_actions[MAX_CLIENTS];
__int64 frequency;

#define LOOKAHEAD_DISTANCE			300

/*
==================
BotResetNodeSwitches
==================
*/
void BotResetNodeSwitches(void) {
	numnodeswitches = 0;
}

/*
==================
BotDumpNodeSwitches
==================
*/
void BotDumpNodeSwitches(bot_state_t *bs) {
	int i;
	char netname[MAX_NETNAME];

	ClientName(bs->client, netname, sizeof(netname));
	BotAI_Print(PRT_MESSAGE, "%s at %1.1f switched more than %d AI nodes\n", netname, FloatTime(), MAX_NODESWITCHES);
	for (i = 0; i < numnodeswitches; i++) {
		BotAI_Print(PRT_MESSAGE, nodeswitch[i]);
	}
	BotAI_Print(PRT_FATAL, "");
}

/*
==================
BotRecordNodeSwitch
==================
*/
void BotRecordNodeSwitch(bot_state_t *bs, char *node, char *str, char *s) {
	char netname[MAX_NETNAME];

	ClientName(bs->client, netname, sizeof(netname));
	Com_sprintf(nodeswitch[numnodeswitches], 144, "%s at %2.1f entered %s: %s from %s\n", netname, FloatTime(), node, str, s);
#ifdef DEBUG
	if (0) {
		BotAI_Print(PRT_MESSAGE, nodeswitch[numnodeswitches]);
	}
#endif //DEBUG
	numnodeswitches++;
}

/*
==================
BotGetAirGoal
==================
*/
int BotGetAirGoal(bot_state_t *bs, bot_goal_t *goal) {
	bsp_trace_t bsptrace;
	vec3_t end, mins = {-15, -15, -2}, maxs = {15, 15, 2};
	int areanum;

	//trace up until we hit solid
	VectorCopy(bs->origin, end);
	end[2] += 1000;
	BotAI_Trace(&bsptrace, bs->origin, mins, maxs, end, bs->entitynum, CONTENTS_SOLID|CONTENTS_PLAYERCLIP);
	//trace down until we hit water
	VectorCopy(bsptrace.endpos, end);
	BotAI_Trace(&bsptrace, end, mins, maxs, bs->origin, bs->entitynum, CONTENTS_WATER|CONTENTS_SLIME|CONTENTS_LAVA);
	//if we found the water surface
	if (bsptrace.fraction > 0) {
		areanum = BotPointAreaNum(bsptrace.endpos);
		if (areanum) {
			VectorCopy(bsptrace.endpos, goal->origin);
			goal->origin[2] -= 2;
			goal->areanum = areanum;
			goal->mins[0] = -15;
			goal->mins[1] = -15;
			goal->mins[2] = -1;
			goal->maxs[0] = 15;
			goal->maxs[1] = 15;
			goal->maxs[2] = 1;
			goal->flags = GFL_AIR;
			goal->number = 0;
			goal->iteminfo = 0;
			goal->entitynum = 0;
			return qtrue;
		}
	}
	return qfalse;
}

/*
==================
BotGoForAir
==================
*/
int BotGoForAir(bot_state_t *bs, int tfl, bot_goal_t *ltg, float range) {
	bot_goal_t goal;

	//if the bot needs air
	if (bs->lastair_time < FloatTime() - 6) {
		//
#ifdef DEBUG
		//BotAI_Print(PRT_MESSAGE, "going for air\n");
#endif //DEBUG
		//if we can find an air goal
		if (BotGetAirGoal(bs, &goal)) {
			trap_BotPushGoal(bs->gs, &goal);
			return qtrue;
		}
		else {
			//get a nearby goal outside the water
			while(trap_BotChooseNBGItem(bs->gs, bs->origin, bs->inventory, tfl, ltg, range, 1)) {
				trap_BotGetTopGoal(bs->gs, &goal);
				//if the goal is not in water
				if (!(trap_AAS_PointContents(goal.origin) & (CONTENTS_WATER|CONTENTS_SLIME|CONTENTS_LAVA))) {
					return qtrue;
				}
				trap_BotPopGoal(bs->gs);
			}
			trap_BotResetAvoidGoals(bs->gs);
		}
	}
	return qfalse;
}

/*
==================
BotNearbyGoal
==================
*/
int BotNearbyGoal(bot_state_t *bs, int tfl, bot_goal_t *ltg, float range, int set_avoid) {
	int ret;

	//check if the bot should go for air
	if (BotGoForAir(bs, tfl, ltg, range)) return qtrue;
	//if the bot is carrying the enemy flag
	if (BotCTFCarryingFlag(bs)) {
		//if the bot is just a few secs away from the base 
		if (trap_AAS_AreaTravelTimeToGoalArea(bs->areanum, bs->origin,
				bs->teamgoal.areanum, TFL_DEFAULT) < 300) {
			//make the range really small
			range = 50;
		}
	}
	//
	ret = trap_BotChooseNBGItem(bs->gs, bs->origin, bs->inventory, tfl, ltg, range, set_avoid);
	/*
	if (ret)
	{
		char buf[128];
		//get the goal at the top of the stack
		trap_BotGetTopGoal(bs->gs, &goal);
		trap_BotGoalName(goal.number, buf, sizeof(buf));
		BotAI_Print(PRT_MESSAGE, "%1.1f: new nearby goal %s\n", FloatTime(), buf);
	}
    */
	return ret;
}

/*
==================
BotReachedGoal
==================
*/
int BotReachedGoal(bot_state_t *bs, bot_goal_t *goal) {
	if (goal->flags & GFL_ITEM) {
		//if touching the goal
		if (trap_BotTouchingGoal(bs->origin, goal)) {
			if (!(goal->flags & GFL_DROPPED)) {
				trap_BotSetAvoidGoalTime(bs->gs, goal->number, -1);
			}
			return qtrue;
		}
		//if the goal isn't there
		if (trap_BotItemGoalInVisButNotVisible(bs->entitynum, bs->eye, bs->viewangles, goal)) {
			/*
			float avoidtime;
			int t;

			avoidtime = trap_BotAvoidGoalTime(bs->gs, goal->number);
			if (avoidtime > 0) {
				t = trap_AAS_AreaTravelTimeToGoalArea(bs->areanum, bs->origin, goal->areanum, bs->tfl);
				if ((float) t * 0.009 < avoidtime)
					return qtrue;
			}
			*/
			return qtrue;
		}
		//if in the goal area and below or above the goal and not swimming
		if (bs->areanum == goal->areanum) {
			if (bs->origin[0] > goal->origin[0] + goal->mins[0] && bs->origin[0] < goal->origin[0] + goal->maxs[0]) {
				if (bs->origin[1] > goal->origin[1] + goal->mins[1] && bs->origin[1] < goal->origin[1] + goal->maxs[1]) {
					if (!trap_AAS_Swimming(bs->origin)) {
						return qtrue;
					}
				}
			}
		}
	}
	else if (goal->flags & GFL_AIR) {
		//if touching the goal
		if (trap_BotTouchingGoal(bs->origin, goal)) return qtrue;
		//if the bot got air
		if (bs->lastair_time > FloatTime() - 1) return qtrue;
	}
	else {
		//if touching the goal
		if (trap_BotTouchingGoal(bs->origin, goal)) return qtrue;
	}
	return qfalse;
}

/*
==================
BotGetItemLongTermGoal
==================
*/
int BotGetItemLongTermGoal(bot_state_t *bs, int tfl, bot_goal_t *goal, int set_avoid) {
	//if the bot has no goal
	if (!trap_BotGetTopGoal(bs->gs, goal)) {
		//BotAI_Print(PRT_MESSAGE, "no ltg on stack\n");
		bs->ltg_time = 0;
	}
	//if the bot touches the current goal
	else if (BotReachedGoal(bs, goal)) {
		BotChooseWeapon(bs);
		bs->ltg_time = 0;
	}
	//if it is time to find a new long term goal
	if (bs->ltg_time < FloatTime()) {
		//pop the current goal from the stack
		trap_BotPopGoal(bs->gs);
		//BotAI_Print(PRT_MESSAGE, "%s: choosing new ltg\n", ClientName(bs->client, netname, sizeof(netname)));
		//choose a new goal
		//BotAI_Print(PRT_MESSAGE, "%6.1f client %d: BotChooseLTGItem\n", FloatTime(), bs->client);
		if (trap_BotChooseLTGItem(bs->gs, bs->origin, bs->inventory, tfl, set_avoid)) {
			/*
			char buf[128];
			//get the goal at the top of the stack
			trap_BotGetTopGoal(bs->gs, goal);
			trap_BotGoalName(goal->number, buf, sizeof(buf));
			BotAI_Print(PRT_MESSAGE, "%1.1f: new long term goal %s\n", FloatTime(), buf);
            */
			bs->ltg_time = FloatTime() + 20;
		}
		else {//the bot gets sorta stuck with all the avoid timings, shouldn't happen though
			//
#ifdef DEBUG
			char netname[128];

			BotAI_Print(PRT_MESSAGE, "%s: no valid ltg (probably stuck)\n", ClientName(bs->client, netname, sizeof(netname)));
#endif
			//trap_BotDumpAvoidGoals(bs->gs);
			//reset the avoid goals and the avoid reach
			trap_BotResetAvoidGoals(bs->gs);
			trap_BotResetAvoidReach(bs->ms);
		}
		//get the goal at the top of the stack
		return trap_BotGetTopGoal(bs->gs, goal);
	}
	return qtrue;
}

/*
==================
BotGetLongTermGoal

we could also create a seperate AI node for every long term goal type
however this saves us a lot of code
==================
*/
int BotGetLongTermGoal(bot_state_t *bs, int tfl, int retreat, bot_goal_t *goal, int set_avoid) {
	vec3_t target, dir, dir2;
	char netname[MAX_NETNAME];
	char buf[MAX_MESSAGE_SIZE];
	int areanum;
	float croucher;
	aas_entityinfo_t entinfo, botinfo;
	bot_waypoint_t *wp;

	if (bs->ltgtype == LTG_TEAMHELP && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "help_start", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_YES);
			trap_EA_Action(bs->client, ACTION_AFFIRMATIVE);
			bs->teammessage_time = 0;
		}
		//if trying to help the team mate for more than a minute
		if (bs->teamgoal_time < FloatTime())
			bs->ltgtype = 0;
		//if the team mate IS visible for quite some time
		if (bs->teammatevisible_time < FloatTime() - 10) bs->ltgtype = 0;
		//get entity information of the companion
		BotEntityInfo(bs->teammate, &entinfo);
		//if the team mate is visible
		if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->teammate)) {
			//if close just stand still there
			VectorSubtract(entinfo.origin, bs->origin, dir);
			if (VectorLengthSquared(dir) < Square(100)) {
				trap_BotResetAvoidReach(bs->ms);
				return qfalse;
			}
		}
		else {
			//last time the bot was NOT visible
			bs->teammatevisible_time = FloatTime();
		}
		//if the entity information is valid (entity in PVS)
		if (entinfo.valid) {
			areanum = BotPointAreaNum(entinfo.origin);
			if (areanum && trap_AAS_AreaReachability(areanum)) {
				//update team goal
				bs->teamgoal.entitynum = bs->teammate;
				bs->teamgoal.areanum = areanum;
				VectorCopy(entinfo.origin, bs->teamgoal.origin);
				VectorSet(bs->teamgoal.mins, -8, -8, -8);
				VectorSet(bs->teamgoal.maxs, 8, 8, 8);
			}
		}
		memcpy(goal, &bs->teamgoal, sizeof(bot_goal_t));
		return qtrue;
	}
	//if the bot accompanies someone
	if (bs->ltgtype == LTG_TEAMACCOMPANY && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "accompany_start", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_YES);
			trap_EA_Action(bs->client, ACTION_AFFIRMATIVE);
			bs->teammessage_time = 0;
		}
		//if accompanying the companion for 3 minutes
		if (bs->teamgoal_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "accompany_stop", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
			trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
			bs->ltgtype = 0;
		}
		//get entity information of the companion
		BotEntityInfo(bs->teammate, &entinfo);
		//if the companion is visible
		if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->teammate)) {
			//update visible time
			bs->teammatevisible_time = FloatTime();
			VectorSubtract(entinfo.origin, bs->origin, dir);
			if (VectorLengthSquared(dir) < Square(bs->formation_dist)) {
				//
				// if the client being followed bumps into this bot then
				// the bot should back up
				BotEntityInfo(bs->entitynum, &botinfo);
				// if the followed client is not standing ontop of the bot
				if (botinfo.origin[2] + botinfo.maxs[2] > entinfo.origin[2] + entinfo.mins[2]) {
					// if the bounding boxes touch each other
					if (botinfo.origin[0] + botinfo.maxs[0] > entinfo.origin[0] + entinfo.mins[0] - 4&&
						botinfo.origin[0] + botinfo.mins[0] < entinfo.origin[0] + entinfo.maxs[0] + 4) {
						if (botinfo.origin[1] + botinfo.maxs[1] > entinfo.origin[1] + entinfo.mins[1] - 4 &&
							botinfo.origin[1] + botinfo.mins[1] < entinfo.origin[1] + entinfo.maxs[1] + 4) {
							if (botinfo.origin[2] + botinfo.maxs[2] > entinfo.origin[2] + entinfo.mins[2] - 4 &&
								botinfo.origin[2] + botinfo.mins[2] < entinfo.origin[2] + entinfo.maxs[2] + 4) {
								// if the followed client looks in the direction of this bot
								AngleVectors(entinfo.angles, dir, NULL, NULL);
								dir[2] = 0;
								VectorNormalize(dir);
								//VectorSubtract(entinfo.origin, entinfo.lastvisorigin, dir);
								VectorSubtract(bs->origin, entinfo.origin, dir2);
								VectorNormalize(dir2);
								if (DotProduct(dir, dir2) > 0.7) {
									// back up
									BotSetupForMovement(bs);
									trap_BotMoveInDirection(bs->ms, dir2, 400, MOVE_WALK);
								}
							}
						}
					}
				}
				//check if the bot wants to crouch
				//don't crouch if crouched less than 5 seconds ago
				if (bs->attackcrouch_time < FloatTime() - 5) {
					croucher = trap_Characteristic_BFloat(bs->character, CHARACTERISTIC_CROUCHER, 0, 1);
					if (random() < bs->thinktime * croucher) {
						bs->attackcrouch_time = FloatTime() + 5 + croucher * 15;
					}
				}
				//don't crouch when swimming
				if (trap_AAS_Swimming(bs->origin)) bs->attackcrouch_time = FloatTime() - 1;
				//if not arrived yet or arived some time ago
				if (bs->arrive_time < FloatTime() - 2) {
					//if not arrived yet
					if (!bs->arrive_time) {
						trap_EA_Gesture(bs->client);
						BotAI_BotInitialChat(bs, "accompany_arrive", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
						trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
						bs->arrive_time = FloatTime();
					}
					//if the bot wants to crouch
					else if (bs->attackcrouch_time > FloatTime()) {
						trap_EA_Crouch(bs->client);
					}
					//else do some model taunts
					else if (random() < bs->thinktime * 0.05) {
						//do a gesture :)
						trap_EA_Gesture(bs->client);
					}
				}
				//if just arrived look at the companion
				if (bs->arrive_time > FloatTime() - 2) {
					VectorSubtract(entinfo.origin, bs->origin, dir);
					vectoangles(dir, bs->ideal_viewangles);
					bs->ideal_viewangles[2] *= 0.5;
				}
				//else look strategically around for enemies
				else if (random() < bs->thinktime * 0.8) {
					BotRoamGoal(bs, target);
					VectorSubtract(target, bs->origin, dir);
					vectoangles(dir, bs->ideal_viewangles);
					bs->ideal_viewangles[2] *= 0.5;
				}
				//check if the bot wants to go for air
				if (BotGoForAir(bs, bs->tfl, &bs->teamgoal, 400)) {
					trap_BotResetLastAvoidReach(bs->ms);
					//get the goal at the top of the stack
					//trap_BotGetTopGoal(bs->gs, &tmpgoal);
					//trap_BotGoalName(tmpgoal.number, buf, 144);
					//BotAI_Print(PRT_MESSAGE, "new nearby goal %s\n", buf);
					//time the bot gets to pick up the nearby goal item
					bs->nbg_time = FloatTime() + 8;
					AIEnter_Seek_NBG(bs, "BotLongTermGoal: go for air");
					return qfalse;
				}
				//
				trap_BotResetAvoidReach(bs->ms);
				return qfalse;
			}
		}
		//if the entity information is valid (entity in PVS)
		if (entinfo.valid) {
			areanum = BotPointAreaNum(entinfo.origin);
			if (areanum && trap_AAS_AreaReachability(areanum)) {
				//update team goal
				bs->teamgoal.entitynum = bs->teammate;
				bs->teamgoal.areanum = areanum;
				VectorCopy(entinfo.origin, bs->teamgoal.origin);
				VectorSet(bs->teamgoal.mins, -8, -8, -8);
				VectorSet(bs->teamgoal.maxs, 8, 8, 8);
			}
		}
		//the goal the bot should go for
		memcpy(goal, &bs->teamgoal, sizeof(bot_goal_t));
		//if the companion is NOT visible for too long
		if (bs->teammatevisible_time < FloatTime() - 60) {
			BotAI_BotInitialChat(bs, "accompany_cannotfind", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
			trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
			bs->ltgtype = 0;
			// just to make sure the bot won't spam this message
			bs->teammatevisible_time = FloatTime();
		}
		return qtrue;
	}
	//
	if (bs->ltgtype == LTG_DEFENDKEYAREA) {
		if (trap_AAS_AreaTravelTimeToGoalArea(bs->areanum, bs->origin,
				bs->teamgoal.areanum, TFL_DEFAULT) > bs->defendaway_range) {
			bs->defendaway_time = 0;
		}
	}
	//if defending a key area
	if (bs->ltgtype == LTG_DEFENDKEYAREA && !retreat &&
				bs->defendaway_time < FloatTime()) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			trap_BotGoalName(bs->teamgoal.number, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "defend_start", buf, NULL);
			trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
			BotVoiceChatOnly(bs, -1, VOICECHAT_ONDEFENSE);
			bs->teammessage_time = 0;
		}
		//set the bot goal
		memcpy(goal, &bs->teamgoal, sizeof(bot_goal_t));
		//stop after 2 minutes
		if (bs->teamgoal_time < FloatTime()) {
			trap_BotGoalName(bs->teamgoal.number, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "defend_stop", buf, NULL);
			trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
			bs->ltgtype = 0;
		}
		//if very close... go away for some time
		VectorSubtract(goal->origin, bs->origin, dir);
		if (VectorLengthSquared(dir) < Square(70)) {
			trap_BotResetAvoidReach(bs->ms);
			bs->defendaway_time = FloatTime() + 3 + 3 * random();
			if (BotHasPersistantPowerupAndWeapon(bs)) {
				bs->defendaway_range = 100;
			}
			else {
				bs->defendaway_range = 350;
			}
		}
		return qtrue;
	}
	//going to kill someone
	if (bs->ltgtype == LTG_KILL && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			EasyClientName(bs->teamgoal.entitynum, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "kill_start", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			bs->teammessage_time = 0;
		}
		//
		if (bs->lastkilledplayer == bs->teamgoal.entitynum) {
			EasyClientName(bs->teamgoal.entitynum, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "kill_done", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			bs->lastkilledplayer = -1;
			bs->ltgtype = 0;
		}
		//
		if (bs->teamgoal_time < FloatTime()) {
			bs->ltgtype = 0;
		}
		//just roam around
		return BotGetItemLongTermGoal(bs, tfl, goal, set_avoid);
	}
	//get an item
	if (bs->ltgtype == LTG_GETITEM && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			trap_BotGoalName(bs->teamgoal.number, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "getitem_start", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_YES);
			trap_EA_Action(bs->client, ACTION_AFFIRMATIVE);
			bs->teammessage_time = 0;
		}
		//set the bot goal
		memcpy(goal, &bs->teamgoal, sizeof(bot_goal_t));
		//stop after some time
		if (bs->teamgoal_time < FloatTime()) {
			bs->ltgtype = 0;
		}
		//
		if (trap_BotItemGoalInVisButNotVisible(bs->entitynum, bs->eye, bs->viewangles, goal)) {
			trap_BotGoalName(bs->teamgoal.number, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "getitem_notthere", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			bs->ltgtype = 0;
		}
		else if (BotReachedGoal(bs, goal)) {
			trap_BotGoalName(bs->teamgoal.number, buf, sizeof(buf));
			BotAI_BotInitialChat(bs, "getitem_gotit", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			bs->ltgtype = 0;
		}
		return qtrue;
	}
	//if camping somewhere
	if ((bs->ltgtype == LTG_CAMP || bs->ltgtype == LTG_CAMPORDER) && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			if (bs->ltgtype == LTG_CAMPORDER) {
				BotAI_BotInitialChat(bs, "camp_start", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
				trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
				BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_YES);
				trap_EA_Action(bs->client, ACTION_AFFIRMATIVE);
			}
			bs->teammessage_time = 0;
		}
		//set the bot goal
		memcpy(goal, &bs->teamgoal, sizeof(bot_goal_t));
		//
		if (bs->teamgoal_time < FloatTime()) {
			if (bs->ltgtype == LTG_CAMPORDER) {
				BotAI_BotInitialChat(bs, "camp_stop", NULL);
				trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			}
			bs->ltgtype = 0;
		}
		//if really near the camp spot
		VectorSubtract(goal->origin, bs->origin, dir);
		if (VectorLengthSquared(dir) < Square(60))
		{
			//if not arrived yet
			if (!bs->arrive_time) {
				if (bs->ltgtype == LTG_CAMPORDER) {
					BotAI_BotInitialChat(bs, "camp_arrive", EasyClientName(bs->teammate, netname, sizeof(netname)), NULL);
					trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
					BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_INPOSITION);
				}
				bs->arrive_time = FloatTime();
			}
			//look strategically around for enemies
			if (random() < bs->thinktime * 0.8) {
				BotRoamGoal(bs, target);
				VectorSubtract(target, bs->origin, dir);
				vectoangles(dir, bs->ideal_viewangles);
				bs->ideal_viewangles[2] *= 0.5;
			}
			//check if the bot wants to crouch
			//don't crouch if crouched less than 5 seconds ago
			if (bs->attackcrouch_time < FloatTime() - 5) {
				croucher = trap_Characteristic_BFloat(bs->character, CHARACTERISTIC_CROUCHER, 0, 1);
				if (random() < bs->thinktime * croucher) {
					bs->attackcrouch_time = FloatTime() + 5 + croucher * 15;
				}
			}
			//if the bot wants to crouch
			if (bs->attackcrouch_time > FloatTime()) {
				trap_EA_Crouch(bs->client);
			}
			//don't crouch when swimming
			if (trap_AAS_Swimming(bs->origin)) bs->attackcrouch_time = FloatTime() - 1;
			//make sure the bot is not gonna drown
			if (trap_PointContents(bs->eye,bs->entitynum) & (CONTENTS_WATER|CONTENTS_SLIME|CONTENTS_LAVA)) {
				if (bs->ltgtype == LTG_CAMPORDER) {
					BotAI_BotInitialChat(bs, "camp_stop", NULL);
					trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
					//
					if (bs->lastgoal_ltgtype == LTG_CAMPORDER) {
						bs->lastgoal_ltgtype = 0;
					}
				}
				bs->ltgtype = 0;
			}
			//
			if (bs->camp_range > 0) {
				//FIXME: move around a bit
			}
			//
			trap_BotResetAvoidReach(bs->ms);
			return qfalse;
		}
		return qtrue;
	}
	//patrolling along several waypoints
	if (bs->ltgtype == LTG_PATROL && !retreat) {
		//check for bot typing status message
		if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
			strcpy(buf, "");
			for (wp = bs->patrolpoints; wp; wp = wp->next) {
				strcat(buf, wp->name);
				if (wp->next) strcat(buf, " to ");
			}
			BotAI_BotInitialChat(bs, "patrol_start", buf, NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			BotVoiceChatOnly(bs, bs->decisionmaker, VOICECHAT_YES);
			trap_EA_Action(bs->client, ACTION_AFFIRMATIVE);
			bs->teammessage_time = 0;
		}
		//
		if (!bs->curpatrolpoint) {
			bs->ltgtype = 0;
			return qfalse;
		}
		//if the bot touches the current goal
		if (trap_BotTouchingGoal(bs->origin, &bs->curpatrolpoint->goal)) {
			if (bs->patrolflags & PATROL_BACK) {
				if (bs->curpatrolpoint->prev) {
					bs->curpatrolpoint = bs->curpatrolpoint->prev;
				}
				else {
					bs->curpatrolpoint = bs->curpatrolpoint->next;
					bs->patrolflags &= ~PATROL_BACK;
				}
			}
			else {
				if (bs->curpatrolpoint->next) {
					bs->curpatrolpoint = bs->curpatrolpoint->next;
				}
				else {
					bs->curpatrolpoint = bs->curpatrolpoint->prev;
					bs->patrolflags |= PATROL_BACK;
				}
			}
		}
		//stop after 5 minutes
		if (bs->teamgoal_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "patrol_stop", NULL);
			trap_BotEnterChat(bs->cs, bs->decisionmaker, CHAT_TELL);
			bs->ltgtype = 0;
		}
		if (!bs->curpatrolpoint) {
			bs->ltgtype = 0;
			return qfalse;
		}
		memcpy(goal, &bs->curpatrolpoint->goal, sizeof(bot_goal_t));
		return qtrue;
	}
#ifdef CTF
	if (gametype == GT_CTF) {
		//if going for enemy flag
		if (bs->ltgtype == LTG_GETFLAG) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "captureflag_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONGETFLAG);
				bs->teammessage_time = 0;
			}
			//
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &ctf_blueflag, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &ctf_redflag, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//if touching the flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				// make sure the bot knows the flag isn't there anymore
				switch(BotTeam(bs)) {
					case TEAM_RED: bs->blueflagstatus = 1; break;
					case TEAM_BLUE: bs->redflagstatus = 1; break;
				}
				bs->ltgtype = 0;
			}
			//stop after 3 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			BotAlternateRoute(bs, goal);
			return qtrue;
		}
		//if rushing to the base
		if (bs->ltgtype == LTG_RUSHBASE && bs->rushbaseaway_time < FloatTime()) {
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &ctf_redflag, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &ctf_blueflag, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//if not carrying the flag anymore
			if (!BotCTFCarryingFlag(bs)) bs->ltgtype = 0;
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) bs->ltgtype = 0;
			//if touching the base flag the bot should loose the enemy flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				//if the bot is still carrying the enemy flag then the
				//base flag is gone, now just walk near the base a bit
				if (BotCTFCarryingFlag(bs)) {
					trap_BotResetAvoidReach(bs->ms);
					bs->rushbaseaway_time = FloatTime() + 5 + 10 * random();
					//FIXME: add chat to tell the others to get back the flag
				}
				else {
					bs->ltgtype = 0;
				}
			}
			BotAlternateRoute(bs, goal);
			return qtrue;
		}
		//returning flag
		if (bs->ltgtype == LTG_RETURNFLAG) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "returnflag_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONRETURNFLAG);
				bs->teammessage_time = 0;
			}
			//
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &ctf_blueflag, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &ctf_redflag, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//if touching the flag
			if (trap_BotTouchingGoal(bs->origin, goal)) bs->ltgtype = 0;
			//stop after 3 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			BotAlternateRoute(bs, goal);
			return qtrue;
		}
	}
#endif //CTF
#ifdef MISSIONPACK
	else if (gametype == GT_1FCTF) {
		if (bs->ltgtype == LTG_GETFLAG) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "captureflag_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONGETFLAG);
				bs->teammessage_time = 0;
			}
			memcpy(goal, &ctf_neutralflag, sizeof(bot_goal_t));
			//if touching the flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->ltgtype = 0;
			}
			//stop after 3 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			return qtrue;
		}
		//if rushing to the base
		if (bs->ltgtype == LTG_RUSHBASE) {
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &ctf_blueflag, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &ctf_redflag, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//if not carrying the flag anymore
			if (!Bot1FCTFCarryingFlag(bs)) {
				bs->ltgtype = 0;
			}
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			//if touching the base flag the bot should loose the enemy flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->ltgtype = 0;
			}
			BotAlternateRoute(bs, goal);
			return qtrue;
		}
		//attack the enemy base
		if (bs->ltgtype == LTG_ATTACKENEMYBASE &&
				bs->attackaway_time < FloatTime()) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "attackenemybase_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONOFFENSE);
				bs->teammessage_time = 0;
			}
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &ctf_blueflag, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &ctf_redflag, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			//if touching the base flag the bot should loose the enemy flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->attackaway_time = FloatTime() + 2 + 5 * random();
			}
			return qtrue;
		}
		//returning flag
		if (bs->ltgtype == LTG_RETURNFLAG) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "returnflag_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONRETURNFLAG);
				bs->teammessage_time = 0;
			}
			//
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			//just roam around
			return BotGetItemLongTermGoal(bs, tfl, goal);
		}
	}
	else if (gametype == GT_OBELISK) {
		if (bs->ltgtype == LTG_ATTACKENEMYBASE &&
				bs->attackaway_time < FloatTime()) {

			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "attackenemybase_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONOFFENSE);
				bs->teammessage_time = 0;
			}
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &blueobelisk, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &redobelisk, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//if the bot no longer wants to attack the obelisk
			if (BotFeelingBad(bs) > 50) {
				return BotGetItemLongTermGoal(bs, tfl, goal);
			}
			//if touching the obelisk
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->attackaway_time = FloatTime() + 3 + 5 * random();
			}
			// or very close to the obelisk
			VectorSubtract(bs->origin, goal->origin, dir);
			if (VectorLengthSquared(dir) < Square(60)) {
				bs->attackaway_time = FloatTime() + 3 + 5 * random();
			}
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			BotAlternateRoute(bs, goal);
			//just move towards the obelisk
			return qtrue;
		}
	}
	else if (gametype == GT_HARVESTER) {
		//if rushing to the base
		if (bs->ltgtype == LTG_RUSHBASE) {
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &blueobelisk, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &redobelisk, sizeof(bot_goal_t)); break;
				default: BotGoHarvest(bs); return qfalse;
			}
			//if not carrying any cubes
			if (!BotHarvesterCarryingCubes(bs)) {
				BotGoHarvest(bs);
				return qfalse;
			}
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) {
				BotGoHarvest(bs);
				return qfalse;
			}
			//if touching the base flag the bot should loose the enemy flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				BotGoHarvest(bs);
				return qfalse;
			}
			BotAlternateRoute(bs, goal);
			return qtrue;
		}
		//attack the enemy base
		if (bs->ltgtype == LTG_ATTACKENEMYBASE &&
				bs->attackaway_time < FloatTime()) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "attackenemybase_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONOFFENSE);
				bs->teammessage_time = 0;
			}
			switch(BotTeam(bs)) {
				case TEAM_RED: memcpy(goal, &blueobelisk, sizeof(bot_goal_t)); break;
				case TEAM_BLUE: memcpy(goal, &redobelisk, sizeof(bot_goal_t)); break;
				default: bs->ltgtype = 0; return qfalse;
			}
			//quit rushing after 2 minutes
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			//if touching the base flag the bot should loose the enemy flag
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->attackaway_time = FloatTime() + 2 + 5 * random();
			}
			return qtrue;
		}
		//harvest cubes
		if (bs->ltgtype == LTG_HARVEST &&
			bs->harvestaway_time < FloatTime()) {
			//check for bot typing status message
			if (bs->teammessage_time && bs->teammessage_time < FloatTime()) {
				BotAI_BotInitialChat(bs, "harvest_start", NULL);
				trap_BotEnterChat(bs->cs, 0, CHAT_TEAM);
				BotVoiceChatOnly(bs, -1, VOICECHAT_ONOFFENSE);
				bs->teammessage_time = 0;
			}
			memcpy(goal, &neutralobelisk, sizeof(bot_goal_t));
			//
			if (bs->teamgoal_time < FloatTime()) {
				bs->ltgtype = 0;
			}
			//
			if (trap_BotTouchingGoal(bs->origin, goal)) {
				bs->harvestaway_time = FloatTime() + 4 + 3 * random();
			}
			return qtrue;
		}
	}
#endif
	//normal goal stuff
	return BotGetItemLongTermGoal(bs, tfl, goal, set_avoid);
}

/*
==================
BotLongTermGoal
==================
*/
int BotLongTermGoal(bot_state_t *bs, int tfl, int retreat, bot_goal_t *goal, int set_avoid) {
	aas_entityinfo_t entinfo;
	char teammate[MAX_MESSAGE_SIZE];
	float squaredist;
	int areanum;
	vec3_t dir;

	//FIXME: also have air long term goals?
	//
	//if the bot is leading someone and not retreating
	if (bs->lead_time > 0 && !retreat) {
		if (bs->lead_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "lead_stop", EasyClientName(bs->lead_teammate, teammate, sizeof(teammate)), NULL);
			trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
			bs->lead_time = 0;
			return BotGetLongTermGoal(bs, tfl, retreat, goal, set_avoid);
		}
		//
		if (bs->leadmessage_time < 0 && -bs->leadmessage_time < FloatTime()) {
			BotAI_BotInitialChat(bs, "followme", EasyClientName(bs->lead_teammate, teammate, sizeof(teammate)), NULL);
			trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
			bs->leadmessage_time = FloatTime();
		}
		//get entity information of the companion
		BotEntityInfo(bs->lead_teammate, &entinfo);
		//
		if (entinfo.valid) {
			areanum = BotPointAreaNum(entinfo.origin);
			if (areanum && trap_AAS_AreaReachability(areanum)) {
				//update team goal
				bs->lead_teamgoal.entitynum = bs->lead_teammate;
				bs->lead_teamgoal.areanum = areanum;
				VectorCopy(entinfo.origin, bs->lead_teamgoal.origin);
				VectorSet(bs->lead_teamgoal.mins, -8, -8, -8);
				VectorSet(bs->lead_teamgoal.maxs, 8, 8, 8);
			}
		}
		//if the team mate is visible
		if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->lead_teammate)) {
			bs->leadvisible_time = FloatTime();
		}
		//if the team mate is not visible for 1 seconds
		if (bs->leadvisible_time < FloatTime() - 1) {
			bs->leadbackup_time = FloatTime() + 2;
		}
		//distance towards the team mate
		VectorSubtract(bs->origin, bs->lead_teamgoal.origin, dir);
		squaredist = VectorLengthSquared(dir);
		//if backing up towards the team mate
		if (bs->leadbackup_time > FloatTime()) {
			if (bs->leadmessage_time < FloatTime() - 20) {
				BotAI_BotInitialChat(bs, "followme", EasyClientName(bs->lead_teammate, teammate, sizeof(teammate)), NULL);
				trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
				bs->leadmessage_time = FloatTime();
			}
			//if very close to the team mate
			if (squaredist < Square(100)) {
				bs->leadbackup_time = 0;
			}
			//the bot should go back to the team mate
			memcpy(goal, &bs->lead_teamgoal, sizeof(bot_goal_t));
			return qtrue;
		}
		else {
			//if quite distant from the team mate
			if (squaredist > Square(500)) {
				if (bs->leadmessage_time < FloatTime() - 20) {
					BotAI_BotInitialChat(bs, "followme", EasyClientName(bs->lead_teammate, teammate, sizeof(teammate)), NULL);
					trap_BotEnterChat(bs->cs, bs->teammate, CHAT_TELL);
					bs->leadmessage_time = FloatTime();
				}
				//look at the team mate
				VectorSubtract(entinfo.origin, bs->origin, dir);
				vectoangles(dir, bs->ideal_viewangles);
				bs->ideal_viewangles[2] *= 0.5;
				//just wait for the team mate
				return qfalse;
			}
		}
	}
	return BotGetLongTermGoal(bs, tfl, retreat, goal, set_avoid);
}

/*
 * Calculates the straight line distance between two points.
 */
float CalculateLineDistance(vec3_t start, vec3_t end)
{
	float dist = pow(start[0] - end[0], 2);
	dist += pow(start[1] - end[1], 2);
	dist += pow(start[2] - end[2], 2);

	return SQRTFAST(dist);
}

/** 
 * Calculates the reward earned for a given (s,a,s') tuple.
 */
float calculateReward(lspi_action_basis_t *final_state)
{
	float r_health = 0.01 * (float)(final_state->health_diff);
	float r_hit = 0.5 * (float)(final_state->hit_count_diff);
	float r_armor = 0.005 * (float)(final_state->armor_diff);
	float r_kill = 2 * (float)(final_state->kill_diff);
	float r_death = -2 * (float)(final_state->death_diff);

	return r_health + r_hit + r_armor + r_kill + r_death - 0.001;
}

void UpdateBasis(bot_state_t *bs)
{
	LARGE_INTEGER before, after;
	aas_entityinfo_t entinfo;
	bot_goal_t cur_goal;
	float reward;

	if(last_action[bs->client] != -1) // If this isn't the first time we should update the "last basis" first
	{
		memcpy(last_basis[bs->client], basis[bs->client], sizeof(lspi_action_basis_t));
	}
	else
	{
		if(bs->bottype == 1)
		{
			fprintf(reward_file[bs->client], "LSPI Bot\n");
		}
		else if(bs->bottype == 2)
		{
			fprintf(reward_file[bs->client], "Gradient Bot\n");
		}
		else
		{
			fprintf(reward_file[bs->client], "Quake Bot\n");
		}
	}

	// First capture necessary information
	if(!BotFindEnemy(bs, -2))
	{ 
		bs->enemy = -1;
	}
	else
	{
		BotEntityInfo(bs->enemy, &entinfo);
	}
	trap_BotGetTopGoal(bs->gs, &cur_goal);

	// Reward items
	basis[bs->client]->kill_diff = bs->num_kills - basis[bs->client]->kills;
	basis[bs->client]->death_diff = bs->num_deaths - basis[bs->client]->deaths;
	basis[bs->client]->health_diff = bs->cur_ps.stats[STAT_HEALTH] - basis[bs->client]->stat_health;
	basis[bs->client]->armor_diff = bs->cur_ps.stats[STAT_ARMOR] - basis[bs->client]->stat_armor;
	basis[bs->client]->hit_count_diff = bs->lasthitcount - basis[bs->client]->last_hit_count;

	// Stats + K&D
	basis[bs->client]->stat_health = bs->cur_ps.stats[STAT_HEALTH];
	basis[bs->client]->stat_armor = bs->cur_ps.stats[STAT_ARMOR];
	basis[bs->client]->stat_max_health = bs->cur_ps.stats[STAT_MAX_HEALTH];
	basis[bs->client]->kills = bs->num_kills;
	basis[bs->client]->deaths = bs->num_deaths;

	// Powerups
	basis[bs->client]->pw_quad = bs->cur_ps.powerups[PW_QUAD];
	basis[bs->client]->pw_battlesuit= bs->cur_ps.powerups[PW_BATTLESUIT];
	basis[bs->client]->pw_haste = bs->cur_ps.powerups[PW_HASTE];
	basis[bs->client]->pw_invis = bs->cur_ps.powerups[PW_INVIS];
	basis[bs->client]->pw_regen = bs->cur_ps.powerups[PW_REGEN];
	basis[bs->client]->pw_flight = bs->cur_ps.powerups[PW_FLIGHT];
	basis[bs->client]->pw_scout = bs->cur_ps.powerups[PW_SCOUT];
	basis[bs->client]->pw_guard = bs->cur_ps.powerups[PW_GUARD];
	basis[bs->client]->pw_doubler = bs->cur_ps.powerups[PW_DOUBLER];
	basis[bs->client]->pw_ammoregen = bs->cur_ps.powerups[PW_AMMOREGEN];
	basis[bs->client]->pw_invulnerability = bs->cur_ps.powerups[PW_INVULNERABILITY];

	// Ammo
	basis[bs->client]->wp_gauntlet = bs->cur_ps.ammo[WP_GAUNTLET];
	basis[bs->client]->wp_machinegun = bs->cur_ps.ammo[WP_MACHINEGUN];
	basis[bs->client]->wp_shotgun = bs->cur_ps.ammo[WP_SHOTGUN];
	basis[bs->client]->wp_grenade_launcher = bs->cur_ps.ammo[WP_GRENADE_LAUNCHER];
	basis[bs->client]->wp_rocket_launcher = bs->cur_ps.ammo[WP_ROCKET_LAUNCHER];
	basis[bs->client]->wp_lightning = bs->cur_ps.ammo[WP_LIGHTNING];
	basis[bs->client]->wp_railgun = bs->cur_ps.ammo[WP_RAILGUN];
	basis[bs->client]->wp_plasmagun = bs->cur_ps.ammo[WP_PLASMAGUN];
	basis[bs->client]->wp_bfg = bs->cur_ps.ammo[WP_BFG];
	basis[bs->client]->wp_grappling_hook = bs->cur_ps.ammo[WP_GRAPPLING_HOOK];

	// Enemy Information
	if(bs->enemy == -1)
	{
		basis[bs->client]->enemy = -1;
		basis[bs->client]->enemy_is_invisible = -1;
		basis[bs->client]->enemy_is_shooting = -1;
	}
	else
	{
		basis[bs->client]->enemy = 1;
		basis[bs->client]->enemy_is_invisible = EntityIsInvisible(&entinfo);
		basis[bs->client]->enemy_is_shooting = EntityIsShooting(&entinfo);
		basis[bs->client]->enemy_weapon = entinfo.weapon;
		basis[bs->client]->enemy_area_num = bs->lastenemyareanum;
	}
	basis[bs->client]->enemyposition_time = bs->enemyposition_time;
	basis[bs->client]->enemy_line_dist = CalculateLineDistance(bs->origin, bs->enemyorigin);
	
	// Goal information
	basis[bs->client]->goal_flags = cur_goal.flags;

	// Area numbers
	basis[bs->client]->current_area_num = bs->areanum;
	basis[bs->client]->goal_area_num = cur_goal.areanum;

	// Misc
	basis[bs->client]->tfl = bs->tfl;
	basis[bs->client]->last_hit_count = bs->lasthitcount;

	if(basis[bs->client]->kill_diff > 0)
	{
		if(bs->bottype == 1)
		{
			BotAI_Print(PRT_MESSAGE, "LSPI Bot kills another one: %d!\n", bs->num_kills);
		}
		else if(bs->bottype == 2)
		{
			BotAI_Print(PRT_MESSAGE, "Gradient Bot kills another one: %d!\n", bs->num_kills);
		}
	}

	// Write down reward info
	reward = calculateReward(basis[bs->client]);
	total_reward[bs->client] += reward;
	total_actions[bs->client] += 1;
	fprintf(reward_file[bs->client], "%d,%f,%f\n", total_actions[bs->client], reward, total_reward[bs->client]);

	if(bs->bottype == 2 && last_action[bs->client] > 0)
	{
		QueryPerformanceCounter(&before);
		LspiBot_GradUpdate(bs->client, last_basis[bs->client], basis[bs->client], last_action[bs->client]);
		QueryPerformanceCounter(&after);

		policy_update_time[bs->client] += (double)(after.QuadPart - before.QuadPart)/frequency;
		total_updates[bs->client] += 1;
	}
}

void SaveSample(bot_state_t *bs, int action)
{
	lspi_action_basis_t *b = basis[bs->client];
	lspi_action_basis_t *l = last_basis[bs->client];

	if(action == -1) // First time, skip it
	{
		return;
	}

	// Action
	fprintf(sample_file[bs->client], "%d,", action);

	/***** BEGIN WRITE LAST STATE *****/

	// For calculated reward
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,", l->kill_diff, l->death_diff, l->health_diff, l->armor_diff, l->hit_count_diff);

	// Note, we don't save kills or deaths, since those are only for calculating diffs

	// Stats
	fprintf(sample_file[bs->client], "%d,%d,%d,", l->stat_health, l->stat_armor, l->stat_max_health);

	// Powerups
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", l->pw_quad, l->pw_battlesuit, l->pw_haste, l->pw_invis, l->pw_regen, l->pw_flight, l->pw_scout, l->pw_guard, l->pw_doubler, l->pw_ammoregen, l->pw_invulnerability);

	// Ammo
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", l->wp_gauntlet, l->wp_machinegun, l->wp_shotgun, l->wp_grenade_launcher, l->wp_rocket_launcher, l->wp_lightning, l->wp_railgun, l->wp_plasmagun, l->wp_bfg, l->wp_grappling_hook);

	// Enemy Info
	fprintf(sample_file[bs->client], "%d,%.8f,%.8f,%d,%d,%d,", l->enemy, l->enemy_line_dist, l->enemyposition_time, l->enemy_is_invisible, l->enemy_is_shooting, l->enemy_weapon);

	// Goal Info
	fprintf(sample_file[bs->client], "%d,%d,", l->goal_flags, l->item_type);

	// Exit Information
	fprintf(sample_file[bs->client], "%d,%d,%d,", l->last_enemy_area_exits, l->goal_area_exits, l->current_area_exits);

	// Area numbers
	fprintf(sample_file[bs->client], "%d,%d,%d,", l->current_area_num, l->goal_area_num, l->enemy_area_num);

	// Misc
	fprintf(sample_file[bs->client], "%d,%d,", l->tfl, l->last_hit_count);

	/***** END WRITE LAST STATE *****/

	/***** BEGIN WRITE STATE *****/

	// For calculated reward
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,", b->kill_diff, b->death_diff, b->health_diff, b->armor_diff, b->hit_count_diff);

	// Note, we don't save kills or deaths, since those are only for calculating diffs

	// Stats
	fprintf(sample_file[bs->client], "%d,%d,%d,", b->stat_health, b->stat_armor, b->stat_max_health);

	// Powerups
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", b->pw_quad, b->pw_battlesuit, b->pw_haste, b->pw_invis, b->pw_regen, b->pw_flight, b->pw_scout, b->pw_guard, b->pw_doubler, b->pw_ammoregen, b->pw_invulnerability);

	// Ammo
	fprintf(sample_file[bs->client], "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", b->wp_gauntlet, b->wp_machinegun, b->wp_shotgun, b->wp_grenade_launcher, b->wp_rocket_launcher, b->wp_lightning, b->wp_railgun, b->wp_plasmagun, b->wp_bfg, b->wp_grappling_hook);

	// Enemy Info
	fprintf(sample_file[bs->client], "%d,%.8f,%.8f,%d,%d,%d,", b->enemy, b->enemy_line_dist, b->enemyposition_time, b->enemy_is_invisible, b->enemy_is_shooting, b->enemy_weapon);

	// Goal Info
	fprintf(sample_file[bs->client], "%d,%d,", b->goal_flags, b->item_type);

	// Exit Information
	fprintf(sample_file[bs->client], "%d,%d,%d,", b->last_enemy_area_exits, b->goal_area_exits, b->current_area_exits);

	// Area numbers
	fprintf(sample_file[bs->client], "%d,%d,%d,", b->current_area_num, b->goal_area_num, b->enemy_area_num);

	// Misc
	fprintf(sample_file[bs->client], "%d,%d\n", b->tfl, b->last_hit_count);

	/***** END WRITE LAST STATE *****/
}

void AI_Init(bot_state_t *bs)
{
	LARGE_INTEGER li, before, after;

	// Perf tracking
	policy_update_time[bs->client] = 0;
	total_updates[bs->client] = 0;
	action_decision_time[bs->client] = 0;
	total_actions[bs->client] = 0;
	
	if(!QueryPerformanceFrequency(&li))
	{
		BotAI_Print(PRT_ERROR, "Failed to query performance frequency.");
	}
	else
	{
		frequency = li.QuadPart;
	}

	basis[bs->client] = malloc(sizeof(lspi_action_basis_t));
	memset(basis[bs->client], 0, sizeof(lspi_action_basis_t));
	if(bs->bottype)
	{
		LspiBot_Init(bs->client);
#ifdef UPDATE_POLICY
		if(bs->bottype == 1)
		{
			QueryPerformanceCounter(&before);
			LspiBot_Update(bs->client, "samples.dat");
			QueryPerformanceCounter(&after);

			policy_update_time[bs->client] += (double)(after.QuadPart - before.QuadPart)/frequency;
			total_updates[bs->client] += 1;
		}
#endif
	}
	action_chosen[bs->client] = -1;
	last_action[bs->client] = -1;
	last_basis[bs->client] = malloc(sizeof(lspi_action_basis_t));
#ifdef COLLECT_SAMPLES
	if(bs->client == 0)
	{
		sample_file[bs->client] = fopen("samples0.dat", "a");
	}
	else if(bs->client == 1)
	{
		sample_file[bs->client] = fopen("samples1.dat", "a");
	}
	else if(bs->client == 2)
	{
		sample_file[bs->client] = fopen("samples2.dat", "a");
	}
#endif

	// Calculating reward
	if(bs->client == 0)
	{
		reward_file[bs->client] = fopen("reward0.dat", "a");
	}
	else if(bs->client == 1)
	{
		reward_file[bs->client] = fopen("reward1.dat", "a");
	}
	else if(bs->client == 2)
	{
		reward_file[bs->client] = fopen("reward2.dat", "a");
	}
}

void AI_Shutdown(bot_state_t *bs)
{
	FILE *perfFile;

#ifdef COLLECT_SAMPLES
	fclose(sample_file[bs->client]);
	free(last_basis[bs->client]);
#endif
	free(basis[bs->client]);
	if(bs->bottype)
	{
		LspiBot_Shutdown(bs->client);

		if(bs->client == 0)
		{
			perfFile = fopen("perf0.dat", "w");
		}
		else if(bs->client == 1)
		{
			perfFile = fopen("perf1.dat", "w");
		}
		else if(bs->client == 2)
		{
			perfFile = fopen("perf2.dat", "w");
		}
		if(total_updates[bs->client] > 0)
		{
			fprintf(perfFile, "Average Policy Update Time: %f\n", 1000.0*(policy_update_time[bs->client]/total_updates[bs->client]));
		}
		fprintf(perfFile, "Average Action Decision Time: %f\n", 1000.0*(action_decision_time[bs->client]/total_actions[bs->client]));
		fclose(perfFile);
	}

	fclose(reward_file[bs->client]);
}

int AIEnter_Next(bot_state_t *bs, char *s, int act)
{
	LARGE_INTEGER before, after;
	int action;

	if(act == -1)
	{
		QueryPerformanceCounter(&before);
		action = LspiBot_GetAction(bs->client, basis[bs->client]);
		QueryPerformanceCounter(&after);

		action_decision_time[bs->client] += (double)(after.QuadPart - before.QuadPart)/frequency;
	}
	else
	{
		action = act;
	}
	switch(action)
	{
	case LSPI_NBG:
		AIEnter_Seek_NBG(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Seek_NBG.\n");
		}
		break;
	case LSPI_FIGHT:
		AIEnter_Battle_Fight(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Battle_Fight.\n");
		}
		break;
	case LSPI_CHASE:
		AIEnter_Battle_Chase(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Battle_Chase.\n");
		}
		break;
	case LSPI_RETREAT:
		AIEnter_Battle_Retreat(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Battle_Retreat.\n");
		}
		break;
	case LSPI_BATTLE_NBG:
		AIEnter_Battle_NBG(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Battle_NBG.\n");
		}
		break;
	case LSPI_LTG:
		AIEnter_Seek_LTG(bs, s);
		if(last_action[bs->client] != action)
		{
			//BotAI_Print(PRT_MESSAGE, "Seek_LTG.\n");
		}
		break;
	default:
		BotAI_Print(PRT_ERROR, "GetAction returned an invalid result.");
		break;
	}

	last_action[bs->client] = action;
	return action;
}

/*
==================
AIEnter_Intermission
==================
*/
void AIEnter_Intermission(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "intermission", "", s);
	//reset the bot state
	BotResetState(bs);
	//check for end level chat
	if (BotChat_EndLevel(bs)) {
		trap_BotEnterChat(bs->cs, 0, bs->chatto);
	}
	bs->ainode = AINode_Intermission;
}

/*
==================
AINode_Intermission
==================
*/
int AINode_Intermission(bot_state_t *bs) {
	//if the intermission ended
	if (!BotIntermission(bs)) {
		if (BotChat_StartLevel(bs)) {
			bs->stand_time = FloatTime() + BotChatTime(bs);
		}
		else {
			bs->stand_time = FloatTime() + 2;
		}
		AIEnter_Stand(bs, "intermission: chat");
	}
	return qtrue;
}

/*
==================
AIEnter_Observer
==================
*/
void AIEnter_Observer(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "observer", "", s);
	//reset the bot state
	BotResetState(bs);
	bs->ainode = AINode_Observer;
}

/*
==================
AINode_Observer
==================
*/
int AINode_Observer(bot_state_t *bs) {
	//if the bot left observer mode
	if (!BotIsObserver(bs)) {
		AIEnter_Stand(bs, "observer: left observer");
	}
	return qtrue;
}

/*
==================
AIEnter_Stand
==================
*/
void AIEnter_Stand(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "stand", "", s);
	bs->standfindenemy_time = FloatTime() + 1;
	bs->ainode = AINode_Stand;
}

/*
==================
AINode_Stand
==================
*/
int AINode_Stand(bot_state_t *bs) {

	//if the bot's health decreased
	if (bs->lastframe_health > bs->inventory[INVENTORY_HEALTH]) {
		if (BotChat_HitTalking(bs)) {
			bs->standfindenemy_time = FloatTime() + BotChatTime(bs) + 0.1;
			bs->stand_time = FloatTime() + BotChatTime(bs) + 0.1;
		}
	}
	if (bs->standfindenemy_time < FloatTime()) {
		if (BotFindEnemy(bs, -1)) {
			AIEnter_Next(bs, "stand: found enemy", LSPI_FIGHT);
			return qfalse;
		}
		bs->standfindenemy_time = FloatTime() + 1;
	}
	// put up chat icon
	trap_EA_Talk(bs->client);
	// when done standing
	if (bs->stand_time < FloatTime()) {
		trap_BotEnterChat(bs->cs, 0, bs->chatto);
		AIEnter_Next(bs, "stand: time out", LSPI_LTG);
		return qfalse;
	}
	//
	return qtrue;
}

/*
==================
AIEnter_Respawn
==================
*/
void AIEnter_Respawn(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "respawn", "", s);
	//reset some states
	trap_BotResetMoveState(bs->ms);
	trap_BotResetGoalState(bs->gs);
	trap_BotResetAvoidGoals(bs->gs);
	trap_BotResetAvoidReach(bs->ms);
	//if the bot wants to chat
	if (BotChat_Death(bs)) {
		bs->respawn_time = FloatTime() + BotChatTime(bs);
		bs->respawnchat_time = FloatTime();
	}
	else {
		bs->respawn_time = FloatTime() + 1 + random();
		bs->respawnchat_time = 0;
	}
	//set respawn state
	bs->respawn_wait = qfalse;
	bs->ainode = AINode_Respawn;
}

/*
==================
AINode_Respawn
==================
*/
int AINode_Respawn(bot_state_t *bs) {
	// if waiting for the actual respawn
	if (bs->respawn_wait) {
		if (!BotIsDead(bs)) {
			// HACK: Add the previous health and armor values to current to mimic losing health and armor when you die
			basis[bs->client]->stat_health = bs->cur_ps.stats[STAT_HEALTH] + basis[bs->client]->stat_health;
			basis[bs->client]->stat_armor = bs->cur_ps.stats[STAT_ARMOR] + basis[bs->client]->stat_armor;
			AIEnter_Seek_LTG(bs, "respawn: respawned");
		}
		else {
			trap_EA_Respawn(bs->client);
		}
	}
	else if (bs->respawn_time < FloatTime()) {
		// wait until respawned
		bs->respawn_wait = qtrue;
		// elementary action respawn
		trap_EA_Respawn(bs->client);
		//
		if (bs->respawnchat_time) {
			trap_BotEnterChat(bs->cs, 0, bs->chatto);
			bs->enemy = -1;
		}
	}
	if (bs->respawnchat_time && bs->respawnchat_time < FloatTime() - 0.5) {
		trap_EA_Talk(bs->client);
	}
	//
	return qtrue;
}

/*
==================
BotSelectActivateWeapon
==================
*/
int BotSelectActivateWeapon(bot_state_t *bs) {
	//
	if (bs->inventory[INVENTORY_MACHINEGUN] > 0 && bs->inventory[INVENTORY_BULLETS] > 0)
		return WEAPONINDEX_MACHINEGUN;
	else if (bs->inventory[INVENTORY_SHOTGUN] > 0 && bs->inventory[INVENTORY_SHELLS] > 0)
		return WEAPONINDEX_SHOTGUN;
	else if (bs->inventory[INVENTORY_PLASMAGUN] > 0 && bs->inventory[INVENTORY_CELLS] > 0)
		return WEAPONINDEX_PLASMAGUN;
	else if (bs->inventory[INVENTORY_LIGHTNING] > 0 && bs->inventory[INVENTORY_LIGHTNINGAMMO] > 0)
		return WEAPONINDEX_LIGHTNING;
#ifdef MISSIONPACK
	else if (bs->inventory[INVENTORY_CHAINGUN] > 0 && bs->inventory[INVENTORY_BELT] > 0)
		return WEAPONINDEX_CHAINGUN;
	else if (bs->inventory[INVENTORY_NAILGUN] > 0 && bs->inventory[INVENTORY_NAILS] > 0)
		return WEAPONINDEX_NAILGUN;
#endif
	else if (bs->inventory[INVENTORY_RAILGUN] > 0 && bs->inventory[INVENTORY_SLUGS] > 0)
		return WEAPONINDEX_RAILGUN;
	else if (bs->inventory[INVENTORY_ROCKETLAUNCHER] > 0 && bs->inventory[INVENTORY_ROCKETS] > 0)
		return WEAPONINDEX_ROCKET_LAUNCHER;
	else if (bs->inventory[INVENTORY_BFG10K] > 0 && bs->inventory[INVENTORY_BFGAMMO] > 0)
		return WEAPONINDEX_BFG;
	else {
		return -1;
	}
}

/*
==================
BotClearPath

 try to deactivate obstacles like proximity mines on the bot's path
==================
*/
void BotClearPath(bot_state_t *bs, bot_moveresult_t *moveresult) {
	int i, bestmine;
	float dist, bestdist;
	vec3_t target, dir;
	bsp_trace_t bsptrace;
	entityState_t state;

	// if there is a dead body wearing kamikze nearby
	if (bs->kamikazebody) {
		// if the bot's view angles and weapon are not used for movement
		if ( !(moveresult->flags & (MOVERESULT_MOVEMENTVIEW | MOVERESULT_MOVEMENTWEAPON)) ) {
			//
			BotAI_GetEntityState(bs->kamikazebody, &state);
			VectorCopy(state.pos.trBase, target);
			target[2] += 8;
			VectorSubtract(target, bs->eye, dir);
			vectoangles(dir, moveresult->ideal_viewangles);
			//
			moveresult->weapon = BotSelectActivateWeapon(bs);
			if (moveresult->weapon == -1) {
				// FIXME: run away!
				moveresult->weapon = 0;
			}
			if (moveresult->weapon) {
				//
				moveresult->flags |= MOVERESULT_MOVEMENTWEAPON | MOVERESULT_MOVEMENTVIEW;
				// if holding the right weapon
				if (bs->cur_ps.weapon == moveresult->weapon) {
					// if the bot is pretty close with it's aim
					if (InFieldOfVision(bs->viewangles, 20, moveresult->ideal_viewangles)) {
						//
						BotAI_Trace(&bsptrace, bs->eye, NULL, NULL, target, bs->entitynum, MASK_SHOT);
						// if the mine is visible from the current position
						if (bsptrace.fraction >= 1.0 || bsptrace.ent == state.number) {
							// shoot at the mine
							trap_EA_Attack(bs->client);
						}
					}
				}
			}
		}
	}
	if (moveresult->flags & MOVERESULT_BLOCKEDBYAVOIDSPOT) {
		bs->blockedbyavoidspot_time = FloatTime() + 5;
	}
	// if blocked by an avoid spot and the view angles and weapon are used for movement
	if (bs->blockedbyavoidspot_time > FloatTime() &&
		!(moveresult->flags & (MOVERESULT_MOVEMENTVIEW | MOVERESULT_MOVEMENTWEAPON)) ) {
		bestdist = 300;
		bestmine = -1;
		for (i = 0; i < bs->numproxmines; i++) {
			BotAI_GetEntityState(bs->proxmines[i], &state);
			VectorSubtract(state.pos.trBase, bs->origin, dir);
			dist = VectorLength(dir);
			if (dist < bestdist) {
				bestdist = dist;
				bestmine = i;
			}
		}
		if (bestmine != -1) {
			//
			// state->generic1 == TEAM_RED || state->generic1 == TEAM_BLUE
			//
			// deactivate prox mines in the bot's path by shooting
			// rockets or plasma cells etc. at them
			BotAI_GetEntityState(bs->proxmines[bestmine], &state);
			VectorCopy(state.pos.trBase, target);
			target[2] += 2;
			VectorSubtract(target, bs->eye, dir);
			vectoangles(dir, moveresult->ideal_viewangles);
			// if the bot has a weapon that does splash damage
			if (bs->inventory[INVENTORY_PLASMAGUN] > 0 && bs->inventory[INVENTORY_CELLS] > 0)
				moveresult->weapon = WEAPONINDEX_PLASMAGUN;
			else if (bs->inventory[INVENTORY_ROCKETLAUNCHER] > 0 && bs->inventory[INVENTORY_ROCKETS] > 0)
				moveresult->weapon = WEAPONINDEX_ROCKET_LAUNCHER;
			else if (bs->inventory[INVENTORY_BFG10K] > 0 && bs->inventory[INVENTORY_BFGAMMO] > 0)
				moveresult->weapon = WEAPONINDEX_BFG;
			else {
				moveresult->weapon = 0;
			}
			if (moveresult->weapon) {
				//
				moveresult->flags |= MOVERESULT_MOVEMENTWEAPON | MOVERESULT_MOVEMENTVIEW;
				// if holding the right weapon
				if (bs->cur_ps.weapon == moveresult->weapon) {
					// if the bot is pretty close with it's aim
					if (InFieldOfVision(bs->viewangles, 20, moveresult->ideal_viewangles)) {
						//
						BotAI_Trace(&bsptrace, bs->eye, NULL, NULL, target, bs->entitynum, MASK_SHOT);
						// if the mine is visible from the current position
						if (bsptrace.fraction >= 1.0 || bsptrace.ent == state.number) {
							// shoot at the mine
							trap_EA_Attack(bs->client);
						}
					}
				}
			}
		}
	}
}

/*
==================
AIEnter_Seek_ActivateEntity
==================
*/
void AIEnter_Seek_ActivateEntity(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "activate entity", "", s);
	bs->ainode = AINode_Seek_ActivateEntity;
}

/*
==================
AINode_Seek_Activate_Entity
==================
*/
int AINode_Seek_ActivateEntity(bot_state_t *bs) {
	bot_goal_t *goal;
	vec3_t target, dir, ideal_viewangles;
	bot_moveresult_t moveresult;
	int targetvisible;
	bsp_trace_t bsptrace;
	aas_entityinfo_t entinfo;

	if (BotIsObserver(bs)) {
		BotClearActivateGoalStack(bs);
		AIEnter_Observer(bs, "active entity: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		BotClearActivateGoalStack(bs);
		AIEnter_Intermission(bs, "activate entity: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		BotClearActivateGoalStack(bs);
		AIEnter_Respawn(bs, "activate entity: bot dead");
		return qfalse;
	}
	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	// if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	// map specific code
	BotMapScripts(bs);
	// no enemy
	bs->enemy = -1;
	// if the bot has no activate goal
	if (!bs->activatestack) {
		BotClearActivateGoalStack(bs);
		AIEnter_Next(bs, "activate entity: no goal", LSPI_NBG);
		return qfalse;
	}
	//
	goal = &bs->activatestack->goal;
	// initialize target being visible to false
	targetvisible = qfalse;
	// if the bot has to shoot at a target to activate something
	if (bs->activatestack->shoot) {
		//
		BotAI_Trace(&bsptrace, bs->eye, NULL, NULL, bs->activatestack->target, bs->entitynum, MASK_SHOT);
		// if the shootable entity is visible from the current position
		if (bsptrace.fraction >= 1.0 || bsptrace.ent == goal->entitynum) {
			targetvisible = qtrue;
			// if holding the right weapon
			if (bs->cur_ps.weapon == bs->activatestack->weapon) {
				VectorSubtract(bs->activatestack->target, bs->eye, dir);
				vectoangles(dir, ideal_viewangles);
				// if the bot is pretty close with it's aim
				if (InFieldOfVision(bs->viewangles, 20, ideal_viewangles)) {
					trap_EA_Attack(bs->client);
				}
			}
		}
	}
	// if the shoot target is visible
	if (targetvisible) {
		// get the entity info of the entity the bot is shooting at
		BotEntityInfo(goal->entitynum, &entinfo);
		// if the entity the bot shoots at moved
		if (!VectorCompare(bs->activatestack->origin, entinfo.origin)) {
#ifdef DEBUG
			BotAI_Print(PRT_MESSAGE, "hit shootable button or trigger\n");
#endif //DEBUG
			bs->activatestack->time = 0;
		}
		// if the activate goal has been activated or the bot takes too long
		if (bs->activatestack->time < FloatTime()) {
			BotPopFromActivateGoalStack(bs);
			// if there are more activate goals on the stack
			if (bs->activatestack) {
				bs->activatestack->time = FloatTime() + 10;
				return qfalse;
			}
			AIEnter_Next(bs, "activate entity: time out", LSPI_NBG);
			return qfalse;
		}
		memset(&moveresult, 0, sizeof(bot_moveresult_t));
	}
	else {
		// if the bot has no goal
		if (!goal) {
			bs->activatestack->time = 0;
		}
		// if the bot does not have a shoot goal
		else if (!bs->activatestack->shoot) {
			//if the bot touches the current goal
			if (trap_BotTouchingGoal(bs->origin, goal)) {
#ifdef DEBUG
				BotAI_Print(PRT_MESSAGE, "touched button or trigger\n");
#endif //DEBUG
				bs->activatestack->time = 0;
			}
		}
		// if the activate goal has been activated or the bot takes too long
		if (bs->activatestack->time < FloatTime()) {
			BotPopFromActivateGoalStack(bs);
			// if there are more activate goals on the stack
			if (bs->activatestack) {
				bs->activatestack->time = FloatTime() + 10;
				return qfalse;
			}
			AIEnter_Next(bs, "activate entity: activated", LSPI_NBG);
			return qfalse;
		}
		//predict obstacles
		if (BotAIPredictObstacles(bs, goal))
			return qfalse;
		//initialize the movement state
		BotSetupForMovement(bs);
		//move towards the goal
		trap_BotMoveToGoal(&moveresult, bs->ms, goal, bs->tfl);
		//if the movement failed
		if (moveresult.failure) {
			//reset the avoid reach, otherwise bot is stuck in current area
			trap_BotResetAvoidReach(bs->ms);
			//
			bs->activatestack->time = 0;
		}
		//check if the bot is blocked
		BotAIBlocked(bs, &moveresult, qtrue);
	}
	//
	BotClearPath(bs, &moveresult);
	// if the bot has to shoot to activate
	if (bs->activatestack->shoot) {
		// if the view angles aren't yet used for the movement
		if (!(moveresult.flags & MOVERESULT_MOVEMENTVIEW)) {
			VectorSubtract(bs->activatestack->target, bs->eye, dir);
			vectoangles(dir, moveresult.ideal_viewangles);
			moveresult.flags |= MOVERESULT_MOVEMENTVIEW;
		}
		// if there's no weapon yet used for the movement
		if (!(moveresult.flags & MOVERESULT_MOVEMENTWEAPON)) {
			moveresult.flags |= MOVERESULT_MOVEMENTWEAPON;
			//
			bs->activatestack->weapon = BotSelectActivateWeapon(bs);
			if (bs->activatestack->weapon == -1) {
				//FIXME: find a decent weapon first
				bs->activatestack->weapon = 0;
			}
			moveresult.weapon = bs->activatestack->weapon;
		}
	}
	// if the ideal view angles are set for movement
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEWSET|MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	// if waiting for something
	else if (moveresult.flags & MOVERESULT_WAITING) {
		if (random() < bs->thinktime * 0.8) {
			BotRoamGoal(bs, target);
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
			bs->ideal_viewangles[2] *= 0.5;
		}
	}
	else if (!(bs->flags & BFL_IDEALVIEWSET)) {
		if (trap_BotMovementViewTarget(bs->ms, goal, bs->tfl, 300, target)) {
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
		}
		else {
			vectoangles(moveresult.movedir, bs->ideal_viewangles);
		}
		bs->ideal_viewangles[2] *= 0.5;
	}
	// if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON)
		bs->weaponnum = moveresult.weapon;
	// if there is an enemy
	if (BotFindEnemy(bs, -1)) {
		if (BotWantsToRetreat(bs)) {
			//keep the current long term goal and retreat
			AIEnter_Next(bs, "activate entity: found enemy", LSPI_BATTLE_NBG);
		}
		else {
			trap_BotResetLastAvoidReach(bs->ms);
			//empty the goal stack
			trap_BotEmptyGoalStack(bs->gs);
			//go fight
			AIEnter_Next(bs, "activate entity: found enemy", LSPI_FIGHT);
		}
		BotClearActivateGoalStack(bs);
	}
	return qtrue;
}

/*
==================
AIEnter_Seek_NBG
==================
*/
void AIEnter_Seek_NBG(bot_state_t *bs, char *s) {
	bot_goal_t goal;
	char buf[144];

	if (trap_BotGetTopGoal(bs->gs, &goal)) {
		trap_BotGoalName(goal.number, buf, 144);
		BotRecordNodeSwitch(bs, "seek NBG", buf, s);
	}
	else {
		BotRecordNodeSwitch(bs, "seek NBG", "no goal", s);
	}
	bs->ainode = AINode_Seek_NBG;
}

/*
==================
AINode_Seek_NBG
==================
*/
int AINode_Seek_NBG(bot_state_t *bs) {
	bot_goal_t goal;
	vec3_t target, dir;
	bot_moveresult_t moveresult;

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "seek nbg: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "seek nbg: intermision");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "seek nbg: bot dead");
		return qfalse;
	}

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		UpdateBasis(bs);
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_NBG:
				break;
			case LSPI_FIGHT:
				trap_BotResetLastAvoidReach(bs->ms);
				//empty the goal stack
				trap_BotEmptyGoalStack(bs->gs);
				return qfalse;
			default:
				trap_BotPopGoal(bs->gs);
				return qfalse;
			}
		}
	}

	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//
	if (BotCanAndWantsToRocketJump(bs)) {
		bs->tfl |= TFL_ROCKETJUMP;
	}
	//map specific code
	BotMapScripts(bs);
	//no enemy
	bs->enemy = -1;
	//if the bot has no goal
	if (!trap_BotGetTopGoal(bs->gs, &goal)) 
	{
		if(bs->bottype)
		{
			AIEnter_Next(bs, "seek nbg: no goal found", LSPI_LTG);
			return qfalse;
		}
		else
		{
			bs->nbg_time = 0;
		}
	}
	else if (BotReachedGoal(bs, &goal)) {
		bs->nbg_time = 0;
		if(bs->bottype)
		{
			trap_BotSetAvoidGoalTime(bs->gs, goal.number, -1);
			trap_BotPopGoal(bs->gs);
		}
	}
	//
	if(!bs->bottype)
	{
		if (bs->nbg_time < FloatTime()) {
			//pop the current goal from the stack
			trap_BotPopGoal(bs->gs);
			//check for new nearby items right away
			//NOTE: we canNOT reset the check_time to zero because it would create an endless loop of node switches
			bs->check_time = FloatTime() + 0.05;

			//go back to seek ltg
			AIEnter_Next(bs, "seek nbg: time out", LSPI_LTG);
			return qfalse;
		}
	}
	//predict obstacles
	if (BotAIPredictObstacles(bs, &goal))
		return qfalse;
	//initialize the movement state
	BotSetupForMovement(bs);
	//move towards the goal
	trap_BotMoveToGoal(&moveresult, bs->ms, &goal, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		bs->nbg_time = 0;
	}
	//check if the bot is blocked
	BotAIBlocked(bs, &moveresult, qtrue);
	//
	BotClearPath(bs, &moveresult);
	//if the viewangles are used for the movement
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEWSET|MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	//if waiting for something
	else if (moveresult.flags & MOVERESULT_WAITING) {
		if (random() < bs->thinktime * 0.8) {
			BotRoamGoal(bs, target);
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
			bs->ideal_viewangles[2] *= 0.5;
		}
	}
	else if (!(bs->flags & BFL_IDEALVIEWSET)) {
		if (!trap_BotGetSecondGoal(bs->gs, &goal)) trap_BotGetTopGoal(bs->gs, &goal);
		if (trap_BotMovementViewTarget(bs->ms, &goal, bs->tfl, 300, target)) {
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
		}
		//FIXME: look at cluster portals?
		else vectoangles(moveresult.movedir, bs->ideal_viewangles);
		bs->ideal_viewangles[2] *= 0.5;
	}
	//if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON) bs->weaponnum = moveresult.weapon;
	//if there is an enemy
	if(!bs->bottype)
	{
		if (BotFindEnemy(bs, -1)) {
			if (BotWantsToRetreat(bs)) {
				//keep the current long term goal and retreat
				AIEnter_Next(bs, "seek nbg: found enemy", LSPI_BATTLE_NBG);
			}
			else {
				trap_BotResetLastAvoidReach(bs->ms);
				//empty the goal stack
				trap_BotEmptyGoalStack(bs->gs);
				//go fight
				AIEnter_Next(bs, "seek nbg: found enemy", LSPI_FIGHT);
			}
		}
	}

	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_NBG;
	return qtrue;
}

/*
==================
AIEnter_Seek_LTG
==================
*/
void AIEnter_Seek_LTG(bot_state_t *bs, char *s) {
	bot_goal_t goal;
	char buf[144];

	if (trap_BotGetTopGoal(bs->gs, &goal)) {
		trap_BotGoalName(goal.number, buf, 144);
		BotRecordNodeSwitch(bs, "seek LTG", buf, s);
	}
	else {
		BotRecordNodeSwitch(bs, "seek LTG", "no goal", s);
	}
	bs->ainode = AINode_Seek_LTG;
}

/*
==================
AINode_Seek_LTG
==================
*/
int AINode_Seek_LTG(bot_state_t *bs)
{
	bot_goal_t goal;
	vec3_t target, dir;
	bot_moveresult_t moveresult;
	int range, goal_found, set_avoid;
	//char buf[128];
	//bot_goal_t tmpgoal;

	if(bs->bottype)
	{
		set_avoid = 0;
	}
	else
	{
		set_avoid = 1;
	}

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "seek ltg: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "seek ltg: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "seek ltg: bot dead");
		return qfalse;
	}
	//
	if (BotChat_Random(bs)) {
		bs->stand_time = FloatTime() + BotChatTime(bs);
		AIEnter_Stand(bs, "seek ltg: random chat");
		return qfalse;
	}

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		trap_BotGetTopGoal(bs->gs, &goal);
		goal_found = BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
		UpdateBasis(bs);
		if(goal_found)
		{
			trap_BotPopGoal(bs->gs);
		}
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_LTG:
				break;
			case LSPI_NBG:
				BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
				return qfalse;
			case LSPI_FIGHT:
				trap_BotResetLastAvoidReach(bs->ms);
				//empty the goal stack
				trap_BotEmptyGoalStack(bs->gs);
				return qfalse;
			default:
				return qfalse;
			}
		}
	}

	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//
	if (BotCanAndWantsToRocketJump(bs)) {
		bs->tfl |= TFL_ROCKETJUMP;
	}
	//map specific code
	BotMapScripts(bs);
	//no enemy
	bs->enemy = -1;
	//
	if (bs->killedenemy_time > FloatTime() - 2) {
		if (random() < bs->thinktime * 1) {
			trap_EA_Gesture(bs->client);
		}
	}

	//if there is an enemy
	if(!bs->bottype)
	{
		if (BotFindEnemy(bs, -1)) {
			if (BotWantsToRetreat(bs)) {
				//keep the current long term goal and retreat
				AIEnter_Next(bs, "seek ltg: found enemy", LSPI_RETREAT);
				return qfalse;
			}
			else {
				trap_BotResetLastAvoidReach(bs->ms);
				//empty the goal stack
				trap_BotEmptyGoalStack(bs->gs);
				//go fight
				AIEnter_Next(bs, "seek ltg: found enemy", LSPI_FIGHT);
				return qfalse;
			}
		}
	}
	//
	BotTeamGoals(bs, qfalse);
	//get the current long term goal
	if (!BotLongTermGoal(bs, bs->tfl, qfalse, &goal, set_avoid)) {
		action_chosen[bs->client] = -1;
		last_action[bs->client] = LSPI_LTG;
		return qtrue;
	}
	else if(bs->bottype && BotReachedGoal(bs, &goal))
	{
		trap_BotSetAvoidGoalTime(bs->gs, goal.number, -1);
	}
	//check for nearby goals periodicly
	if (bs->check_time < FloatTime()) {
		bs->check_time = FloatTime() + 0.5;
		//check if the bot wants to camp
		BotWantsToCamp(bs);
		//
		if (bs->ltgtype == LTG_DEFENDKEYAREA) range = 400;
		else range = 150;
		//
#ifdef CTF
		if (gametype == GT_CTF) {
			//if carrying a flag the bot shouldn't be distracted too much
			if (BotCTFCarryingFlag(bs))
				range = 50;
		}
#endif //CTF
#ifdef MISSIONPACK
		else if (gametype == GT_1FCTF) {
			if (Bot1FCTFCarryingFlag(bs))
				range = 50;
		}
		else if (gametype == GT_HARVESTER) {
			if (BotHarvesterCarryingCubes(bs))
				range = 80;
		}
#endif
		//
		if(!bs->bottype)
		{
			if (BotNearbyGoal(bs, bs->tfl, &goal, range, 1)) {
				trap_BotResetLastAvoidReach(bs->ms);
				//get the goal at the top of the stack
				//trap_BotGetTopGoal(bs->gs, &tmpgoal);
				//trap_BotGoalName(tmpgoal.number, buf, 144);
				//BotAI_Print(PRT_MESSAGE, "new nearby goal %s\n", buf);
				//time the bot gets to pick up the nearby goal item
				bs->nbg_time = FloatTime() + 4 + range * 0.01;
				AIEnter_Next(bs, "ltg seek: nbg", LSPI_NBG);
				return qfalse;
			}
		}
	}
	//predict obstacles
	if (BotAIPredictObstacles(bs, &goal))
		return qfalse;
	//initialize the movement state
	BotSetupForMovement(bs);
	//move towards the goal
	trap_BotMoveToGoal(&moveresult, bs->ms, &goal, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		//BotAI_Print(PRT_MESSAGE, "movement failure %d\n", moveresult.traveltype);
		bs->ltg_time = 0;
	}
	//
	BotAIBlocked(bs, &moveresult, qtrue);
	//
	BotClearPath(bs, &moveresult);
	//if the viewangles are used for the movement
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEWSET|MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	//if waiting for something
	else if (moveresult.flags & MOVERESULT_WAITING) {
		if (random() < bs->thinktime * 0.8) {
			BotRoamGoal(bs, target);
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
			bs->ideal_viewangles[2] *= 0.5;
		}
	}
	else if (!(bs->flags & BFL_IDEALVIEWSET)) {
		if (trap_BotMovementViewTarget(bs->ms, &goal, bs->tfl, 300, target)) {
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
		}
		//FIXME: look at cluster portals?
		else if (VectorLengthSquared(moveresult.movedir)) {
			vectoangles(moveresult.movedir, bs->ideal_viewangles);
		}
		else if (random() < bs->thinktime * 0.8) {
			BotRoamGoal(bs, target);
			VectorSubtract(target, bs->origin, dir);
			vectoangles(dir, bs->ideal_viewangles);
			bs->ideal_viewangles[2] *= 0.5;
		}
		bs->ideal_viewangles[2] *= 0.5;
	}
	//if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON) bs->weaponnum = moveresult.weapon;
	//

	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_LTG;
	return qtrue;
}

/*
==================
AIEnter_Battle_Fight
==================
*/
void AIEnter_Battle_Fight(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "battle fight", "", s);
	trap_BotResetLastAvoidReach(bs->ms);
	bs->ainode = AINode_Battle_Fight;
}

/*
==================
AIEnter_Battle_Fight
==================
*/
void AIEnter_Battle_SuicidalFight(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "battle fight", "", s);
	trap_BotResetLastAvoidReach(bs->ms);
	bs->ainode = AINode_Battle_Fight;
	bs->flags |= BFL_FIGHTSUICIDAL;
}

/*
==================
AINode_Battle_Fight
==================
*/
int AINode_Battle_Fight(bot_state_t *bs) {
	bot_goal_t goal;
	int areanum, goal_found;
	vec3_t target;
	aas_entityinfo_t entinfo;
	bot_moveresult_t moveresult;

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "battle fight: observer");
		return qfalse;
	}

	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "battle fight: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "battle fight: bot dead");
		return qfalse;
	}

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		trap_BotGetTopGoal(bs->gs, &goal);
		goal_found = BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
		UpdateBasis(bs);
		if(goal_found)
		{
			trap_BotPopGoal(bs->gs);
		}
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_BATTLE_NBG:
				BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
				return qfalse;
			case LSPI_FIGHT:
				break;
			default:
				return qfalse;
			}
		}
	}

	if(!bs->bottype)
	{
		//if there is another better enemy
		if (BotFindEnemy(bs, bs->enemy)) {
	#ifdef DEBUG
			BotAI_Print(PRT_MESSAGE, "fight: found new better enemy\n");
	#endif
		}
		//if no enemy
	
		if (bs->enemy < 0) {
			AIEnter_Next(bs, "battle fight: no enemy", LSPI_LTG);
			return qfalse;
		}
		//
		BotEntityInfo(bs->enemy, &entinfo);
		//if the enemy is dead
		if (bs->enemydeath_time) {
			if (bs->enemydeath_time < FloatTime() - 1.0) {
				bs->enemydeath_time = 0;
				if (bs->enemysuicide) {
					BotChat_EnemySuicide(bs);
				}
				if (bs->lastkilledplayer == bs->enemy && BotChat_Kill(bs)) {
					bs->stand_time = FloatTime() + BotChatTime(bs);
					AIEnter_Stand(bs, "battle fight: enemy dead");
				}
				else {
					bs->ltg_time = 0;
					AIEnter_Next(bs, "battle fight: enemy dead", LSPI_LTG);
				}
				return qfalse;
			}
		}
		else {
			if (EntityIsDead(&entinfo)) {
				bs->enemydeath_time = FloatTime();
			}
		}
		//if the enemy is invisible and not shooting the bot looses track easily
		if (EntityIsInvisible(&entinfo) && !EntityIsShooting(&entinfo)) {
			if (random() < 0.2) {
				AIEnter_Next(bs, "battle fight: invisible", LSPI_LTG);
				return qfalse;
			}
		}
	}
	//
	VectorCopy(entinfo.origin, target);
	// if not a player enemy
	if (bs->enemy >= MAX_CLIENTS) {
#ifdef MISSIONPACK
		// if attacking an obelisk
		if ( bs->enemy == redobelisk.entitynum ||
			bs->enemy == blueobelisk.entitynum ) {
			target[2] += 16;
		}
#endif
	}
	//update the reachability area and origin if possible
	areanum = BotPointAreaNum(target);
	if (areanum && trap_AAS_AreaReachability(areanum)) {
		VectorCopy(target, bs->lastenemyorigin);
		bs->lastenemyareanum = areanum;
	}
	//update the attack inventory values
	BotUpdateBattleInventory(bs, bs->enemy);
	//if the bot's health decreased
	if (bs->lastframe_health > bs->inventory[INVENTORY_HEALTH]) {
		if (BotChat_HitNoDeath(bs)) {
			bs->stand_time = FloatTime() + BotChatTime(bs);
			AIEnter_Stand(bs, "battle fight: chat health decreased");
			return qfalse;
		}
	}
	//if the bot hit someone
	if (bs->cur_ps.persistant[PERS_HITS] > bs->lasthitcount) {
		if (BotChat_HitNoKill(bs)) {
			bs->stand_time = FloatTime() + BotChatTime(bs);
			AIEnter_Stand(bs, "battle fight: chat hit someone");
			return qfalse;
		}
	}
	if(!bs->bottype)
	{
		//if the enemy is not visible
		if (!BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->enemy)) {
			if (BotWantsToChase(bs)) {
				AIEnter_Next(bs, "battle fight: enemy out of sight", LSPI_CHASE);
				return qfalse;
			}
			else
			{
				AIEnter_Next(bs, "battle fight: enemy out of sight", LSPI_LTG);
				return qfalse;
			}
		}
	}
	//use holdable items
	BotBattleUseItems(bs);
	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//
	if (BotCanAndWantsToRocketJump(bs)) {
		bs->tfl |= TFL_ROCKETJUMP;
	}
	//choose the best weapon to fight with
	BotChooseWeapon(bs);
	//do attack movements
	moveresult = BotAttackMove(bs, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		//BotAI_Print(PRT_MESSAGE, "movement failure %d\n", moveresult.traveltype);
		bs->ltg_time = 0;
	}
	//
	BotAIBlocked(bs, &moveresult, qfalse);
	//aim at the enemy
	BotAimAtEnemy(bs);
	//attack the enemy if possible
	BotCheckAttack(bs);
	//if the bot wants to retreat
	if (!(bs->flags & BFL_FIGHTSUICIDAL)) {
		if(!bs->bottype)
		{
			if (BotWantsToRetreat(bs)) {
				AIEnter_Next(bs, "battle fight: wants to retreat", LSPI_RETREAT);
			}
		}
	}
	
	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_FIGHT;
	return qtrue;
}

/*
==================
AIEnter_Battle_Chase
==================
*/
void AIEnter_Battle_Chase(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "battle chase", "", s);
	bs->chase_time = FloatTime();
	bs->ainode = AINode_Battle_Chase;
}

/*
==================
AINode_Battle_Chase
==================
*/
int AINode_Battle_Chase(bot_state_t *bs)
{
	bot_goal_t goal;
	vec3_t target, dir;
	bot_moveresult_t moveresult;
	float range;
	int goal_found;

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "battle chase: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "battle chase: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "battle chase: bot dead");
		return qfalse;
	}

	//create the chase goal
	goal.entitynum = bs->enemy;
	goal.areanum = bs->lastenemyareanum;
	VectorCopy(bs->lastenemyorigin, goal.origin);
	VectorSet(goal.mins, -8, -8, -8);
	VectorSet(goal.maxs, 8, 8, 8);

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		goal_found = BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
		UpdateBasis(bs);
		if(goal_found)
		{
			trap_BotPopGoal(bs->gs);
		}
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_CHASE:
				break;
			case LSPI_BATTLE_NBG:
				trap_BotResetLastAvoidReach(bs->ms);
				BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
				return qfalse;
			default:
				return qfalse;
			}
		}
	}

	if(!bs->bottype)
	{
		//if no enemy
		if (bs->enemy < 0) {
			AIEnter_Next(bs, "battle chase: no enemy", LSPI_LTG);
			return qfalse;
		}
		//if the enemy is visible
		if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->enemy)) {
			AIEnter_Next(bs, "battle chase", LSPI_FIGHT);
			return qfalse;
		}
		//if there is another enemy
		if (BotFindEnemy(bs, -1)) {
			AIEnter_Next(bs, "battle chase: better enemy", LSPI_FIGHT);
			return qfalse;
		}
		//there is no last enemy area
		if (!bs->lastenemyareanum) {
			AIEnter_Next(bs, "battle chase: no enemy area", LSPI_LTG);
			return qfalse;
		}
	}
	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//
	if (BotCanAndWantsToRocketJump(bs)) {
		bs->tfl |= TFL_ROCKETJUMP;
	}
	//map specific code
	BotMapScripts(bs);
	//if the last seen enemy spot is reached the enemy could not be found
	if (trap_BotTouchingGoal(bs->origin, &goal)) bs->chase_time = 0;
	//if there's no chase time left
	if(!bs->bottype)
	{
		if (!bs->chase_time || bs->chase_time < FloatTime() - 10) {
			AIEnter_Next(bs, "battle chase: time out", LSPI_LTG);
			return qfalse;
		}
	//check for nearby goals periodicly
		if (bs->check_time < FloatTime()) {
			bs->check_time = FloatTime() + 1;
			range = 150;
			//
			if (BotNearbyGoal(bs, bs->tfl, &goal, range, 1)) {
				//the bot gets 5 seconds to pick up the nearby goal item
				bs->nbg_time = FloatTime() + 0.1 * range + 1;
				trap_BotResetLastAvoidReach(bs->ms);
				AIEnter_Next(bs, "battle chase: nbg", LSPI_BATTLE_NBG);
				return qfalse;
			}
		}
	}
	//
	BotUpdateBattleInventory(bs, bs->enemy);
	//initialize the movement state
	BotSetupForMovement(bs);
	//move towards the goal
	trap_BotMoveToGoal(&moveresult, bs->ms, &goal, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		//BotAI_Print(PRT_MESSAGE, "movement failure %d\n", moveresult.traveltype);
		bs->ltg_time = 0;
	}
	//
	BotAIBlocked(bs, &moveresult, qfalse);
	//
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEWSET|MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	else if (!(bs->flags & BFL_IDEALVIEWSET)) {
		if (bs->chase_time > FloatTime() - 2) {
			BotAimAtEnemy(bs);
		}
		else {
			if (trap_BotMovementViewTarget(bs->ms, &goal, bs->tfl, 300, target)) {
				VectorSubtract(target, bs->origin, dir);
				vectoangles(dir, bs->ideal_viewangles);
			}
			else {
				vectoangles(moveresult.movedir, bs->ideal_viewangles);
			}
		}
		bs->ideal_viewangles[2] *= 0.5;
	}
	//if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON) bs->weaponnum = moveresult.weapon;
	//if the bot is in the area the enemy was last seen in
	if (bs->areanum == bs->lastenemyareanum) bs->chase_time = 0;
	//if the bot wants to retreat (the bot could have been damage during the chase)
	if(!bs->bottype)
	{
		if (BotWantsToRetreat(bs)) {
			AIEnter_Next(bs, "battle chase: wants to retreat", LSPI_RETREAT);
		}
	}

	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_CHASE;
	return qtrue;
}

/*
==================
AIEnter_Battle_Retreat
==================
*/
void AIEnter_Battle_Retreat(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "battle retreat", "", s);
	bs->ainode = AINode_Battle_Retreat;
}

/*
==================
AINode_Battle_Retreat
==================
*/
int AINode_Battle_Retreat(bot_state_t *bs) {
	bot_goal_t goal;
	aas_entityinfo_t entinfo;
	bot_moveresult_t moveresult;
	vec3_t target, dir;
	float attack_skill, range;
	int areanum, goal_found, set_avoid;

	if(bs->bottype)
	{
		set_avoid = 0;
	}
	else
	{
		set_avoid = 1;
	}

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "battle retreat: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "battle retreat: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "battle retreat: bot dead");
		return qfalse;
	}

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		trap_BotGetTopGoal(bs->gs, &goal);
		goal_found = BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
		UpdateBasis(bs);
		if(goal_found)
		{
			trap_BotPopGoal(bs->gs);
		}
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_RETREAT:
				break;
			case LSPI_CHASE:
				trap_BotEmptyGoalStack(bs->gs);
				return qfalse;
			case LSPI_BATTLE_NBG:
				trap_BotResetLastAvoidReach(bs->ms);
				BotNearbyGoal(bs, bs->tfl, &goal, 400, 0);
				return qfalse;
			default:
				return qfalse;
			}
		}
	}

	if(!bs->bottype)
	{
		if (bs->enemy < 0) {
			AIEnter_Next(bs, "battle retreat: no enemy", LSPI_LTG);
			return qfalse;
		}
		//
		BotEntityInfo(bs->enemy, &entinfo);
		if (EntityIsDead(&entinfo)) {
			AIEnter_Next(bs, "battle retreat: enemy dead", LSPI_LTG);
			return qfalse;
		}
	}
	//if there is another better enemy
	if (BotFindEnemy(bs, bs->enemy)) {
#ifdef DEBUG
		BotAI_Print(PRT_MESSAGE, "retreat: found new better enemy\n");
#endif
	}
	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//map specific code
	BotMapScripts(bs);
	//update the attack inventory values
	BotUpdateBattleInventory(bs, bs->enemy);
	//if the bot doesn't want to retreat anymore... probably picked up some nice items
	if(!bs->bottype)
	{
		if (BotWantsToChase(bs)) {
			//empty the goal stack, when chasing, only the enemy is the goal
			trap_BotEmptyGoalStack(bs->gs);
			//go chase the enemy
			AIEnter_Next(bs, "battle retreat: wants to chase", LSPI_CHASE);
			return qfalse;
		}
	}
	//update the last time the enemy was visible
	if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->enemy)) {
		bs->enemyvisible_time = FloatTime();
		VectorCopy(entinfo.origin, target);
		// if not a player enemy
		if (bs->enemy >= MAX_CLIENTS) {
#ifdef MISSIONPACK
			// if attacking an obelisk
			if ( bs->enemy == redobelisk.entitynum ||
				bs->enemy == blueobelisk.entitynum ) {
				target[2] += 16;
			}
#endif
		}
		//update the reachability area and origin if possible
		areanum = BotPointAreaNum(target);
		if (areanum && trap_AAS_AreaReachability(areanum)) {
			VectorCopy(target, bs->lastenemyorigin);
			bs->lastenemyareanum = areanum;
		}
	}
	//if the enemy is NOT visible for 4 seconds
	if(!bs->bottype)
	{
		if (bs->enemyvisible_time < FloatTime() - 4) {
			AIEnter_Next(bs, "battle retreat: lost enemy", LSPI_LTG);
			return qfalse;
		}
		//else if the enemy is NOT visible
		else if (bs->enemyvisible_time < FloatTime()) {
			//if there is another enemy
			if (BotFindEnemy(bs, -1)) {
				AIEnter_Next(bs, "battle retreat: another enemy", LSPI_FIGHT);
				return qfalse;
			}
		}
	}
	//
	BotTeamGoals(bs, qtrue);
	//use holdable items
	BotBattleUseItems(bs);
	//get the current long term goal while retreating
	if (!BotLongTermGoal(bs, bs->tfl, qtrue, &goal, set_avoid)) {
		AIEnter_Battle_SuicidalFight(bs, "battle retreat: no way out");
		return qfalse;
	}
	if(!bs->bottype)
	{
		//check for nearby goals periodicly
		if (bs->check_time < FloatTime()) {
			bs->check_time = FloatTime() + 1;
			range = 150;
	#ifdef CTF
			if (gametype == GT_CTF) {
				//if carrying a flag the bot shouldn't be distracted too much
				if (BotCTFCarryingFlag(bs))
					range = 50;
			}
	#endif //CTF
	#ifdef MISSIONPACK
			else if (gametype == GT_1FCTF) {
				if (Bot1FCTFCarryingFlag(bs))
					range = 50;
			}
			else if (gametype == GT_HARVESTER) {
				if (BotHarvesterCarryingCubes(bs))
					range = 80;
			}
	#endif
			//
			if (BotNearbyGoal(bs, bs->tfl, &goal, range, 1)) {
				trap_BotResetLastAvoidReach(bs->ms);
				//time the bot gets to pick up the nearby goal item
				bs->nbg_time = FloatTime() + range / 100 + 1;
				AIEnter_Next(bs, "battle retreat: nbg", LSPI_BATTLE_NBG);
				return qfalse;
			}
		}
	}
	//initialize the movement state
	BotSetupForMovement(bs);
	//move towards the goal
	trap_BotMoveToGoal(&moveresult, bs->ms, &goal, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		//BotAI_Print(PRT_MESSAGE, "movement failure %d\n", moveresult.traveltype);
		bs->ltg_time = 0;
	}
	//
	BotAIBlocked(bs, &moveresult, qfalse);
	//choose the best weapon to fight with
	BotChooseWeapon(bs);
	//if the view is fixed for the movement
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	else if (!(moveresult.flags & MOVERESULT_MOVEMENTVIEWSET)
				&& !(bs->flags & BFL_IDEALVIEWSET) ) {
		attack_skill = trap_Characteristic_BFloat(bs->character, CHARACTERISTIC_ATTACK_SKILL, 0, 1);
		//if the bot is skilled anough
		if (attack_skill > 0.3) {
			BotAimAtEnemy(bs);
		}
		else {
			if (trap_BotMovementViewTarget(bs->ms, &goal, bs->tfl, 300, target)) {
				VectorSubtract(target, bs->origin, dir);
				vectoangles(dir, bs->ideal_viewangles);
			}
			else {
				vectoangles(moveresult.movedir, bs->ideal_viewangles);
			}
			bs->ideal_viewangles[2] *= 0.5;
		}
	}
	//if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON) bs->weaponnum = moveresult.weapon;
	//attack the enemy if possible
	BotCheckAttack(bs);
	//

	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_RETREAT;
	return qtrue;
}

/*
==================
AIEnter_Battle_NBG
==================
*/
void AIEnter_Battle_NBG(bot_state_t *bs, char *s) {
	BotRecordNodeSwitch(bs, "battle NBG", "", s);
	bs->ainode = AINode_Battle_NBG;
}

/*
==================
AINode_Battle_NBG
==================
*/
int AINode_Battle_NBG(bot_state_t *bs) {
	int areanum;
	bot_goal_t goal;
	aas_entityinfo_t entinfo;
	bot_moveresult_t moveresult;
	float attack_skill;
	vec3_t target, dir;

	if (BotIsObserver(bs)) {
		AIEnter_Observer(bs, "battle nbg: observer");
		return qfalse;
	}
	//if in the intermission
	if (BotIntermission(bs)) {
		AIEnter_Intermission(bs, "battle nbg: intermission");
		return qfalse;
	}
	//respawn if dead
	if (BotIsDead(bs)) {
		AIEnter_Respawn(bs, "battle nbg: bot dead");
		return qfalse;
	}

	if(action_chosen[bs->client] < 0)
	{
		action_chosen[bs->client] = 1;
		UpdateBasis(bs);
	#ifdef COLLECT_SAMPLES
		SaveSample(bs, last_action[bs->client]);
	#endif

		if(bs->bottype)
		{
			switch(AIEnter_Next(bs, "lspi: finding current action", -1))
			{
			case LSPI_BATTLE_NBG:
				break;
			case LSPI_CHASE:
				trap_BotEmptyGoalStack(bs->gs);
				return qfalse;
			default:
				trap_BotPopGoal(bs->gs);
				return qfalse;
			}
		}
	}

	if(!bs->bottype)
	{
		//if no enemy
		if (bs->enemy < 0) {
			AIEnter_Next(bs, "battle nbg: no enemy", LSPI_NBG);
			return qfalse;
		}
		//
		BotEntityInfo(bs->enemy, &entinfo);
		if (EntityIsDead(&entinfo)) {
			AIEnter_Next(bs, "battle nbg: enemy dead", LSPI_NBG);
			return qfalse;
		}
	}
	//
	bs->tfl = TFL_DEFAULT;
	if (bot_grapple.integer) bs->tfl |= TFL_GRAPPLEHOOK;
	//if in lava or slime the bot should be able to get out
	if (BotInLavaOrSlime(bs)) bs->tfl |= TFL_LAVA|TFL_SLIME;
	//
	if (BotCanAndWantsToRocketJump(bs)) {
		bs->tfl |= TFL_ROCKETJUMP;
	}
	//map specific code
	BotMapScripts(bs);
	//update the last time the enemy was visible
	if (BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->enemy)) {
		bs->enemyvisible_time = FloatTime();
		VectorCopy(entinfo.origin, target);
		// if not a player enemy
		if (bs->enemy >= MAX_CLIENTS) {
#ifdef MISSIONPACK
			// if attacking an obelisk
			if ( bs->enemy == redobelisk.entitynum ||
				bs->enemy == blueobelisk.entitynum ) {
				target[2] += 16;
			}
#endif
		}
		//update the reachability area and origin if possible
		areanum = BotPointAreaNum(target);
		if (areanum && trap_AAS_AreaReachability(areanum)) {
			VectorCopy(target, bs->lastenemyorigin);
			bs->lastenemyareanum = areanum;
		}
	}
	//if the bot has no goal or touches the current goal
	if (!trap_BotGetTopGoal(bs->gs, &goal)) 
	{
		if(bs->bottype)
		{
			AIEnter_Next(bs, "seek nbg: no goal found", LSPI_FIGHT);
			return qfalse;
		}
		else
		{
			bs->nbg_time = 0;
		}
	}
	else if (BotReachedGoal(bs, &goal)) {
		bs->nbg_time = 0;
		if(bs->bottype)
		{
			trap_BotSetAvoidGoalTime(bs->gs, goal.number, -1);
			trap_BotPopGoal(bs->gs);
		}
	}
	//

	if(!bs->bottype)
	{
		if (bs->nbg_time < FloatTime()) {
			trap_BotPopGoal(bs->gs);
			if (trap_BotGetTopGoal(bs->gs, &goal))
				AIEnter_Next(bs, "battle nbg: time out", LSPI_RETREAT);
			else
				AIEnter_Next(bs, "battle nbg: time out", LSPI_FIGHT);
			return qfalse;
		}
	}
	//initialize the movement state
	BotSetupForMovement(bs);
	//move towards the goal
	trap_BotMoveToGoal(&moveresult, bs->ms, &goal, bs->tfl);
	//if the movement failed
	if (moveresult.failure) {
		//reset the avoid reach, otherwise bot is stuck in current area
		trap_BotResetAvoidReach(bs->ms);
		//BotAI_Print(PRT_MESSAGE, "movement failure %d\n", moveresult.traveltype);
		bs->nbg_time = 0;
	}
	//
	BotAIBlocked(bs, &moveresult, qfalse);
	//update the attack inventory values
	BotUpdateBattleInventory(bs, bs->enemy);
	//choose the best weapon to fight with
	BotChooseWeapon(bs);
	//if the view is fixed for the movement
	if (moveresult.flags & (MOVERESULT_MOVEMENTVIEW|MOVERESULT_SWIMVIEW)) {
		VectorCopy(moveresult.ideal_viewangles, bs->ideal_viewangles);
	}
	else if (!(moveresult.flags & MOVERESULT_MOVEMENTVIEWSET)
				&& !(bs->flags & BFL_IDEALVIEWSET)) {
		attack_skill = trap_Characteristic_BFloat(bs->character, CHARACTERISTIC_ATTACK_SKILL, 0, 1);
		//if the bot is skilled anough and the enemy is visible
		if (attack_skill > 0.3) {
			//&& BotEntityVisible(bs->entitynum, bs->eye, bs->viewangles, 360, bs->enemy)
			BotAimAtEnemy(bs);
		}
		else {
			if (trap_BotMovementViewTarget(bs->ms, &goal, bs->tfl, 300, target)) {
				VectorSubtract(target, bs->origin, dir);
				vectoangles(dir, bs->ideal_viewangles);
			}
			else {
				vectoangles(moveresult.movedir, bs->ideal_viewangles);
			}
			bs->ideal_viewangles[2] *= 0.5;
		}
	}
	//if the weapon is used for the bot movement
	if (moveresult.flags & MOVERESULT_MOVEMENTWEAPON) bs->weaponnum = moveresult.weapon;
	//attack the enemy if possible
	BotCheckAttack(bs);
	//

	action_chosen[bs->client] = -1;
	last_action[bs->client] = LSPI_BATTLE_NBG;
	return qtrue;
}

