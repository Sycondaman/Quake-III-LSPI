# Quake-III-LSPI
A modification to Quake III to support an LSPI bot.

Before starting work on Quake III LSPI you should first clone and build the Quake III repository: https://github.com/id-Software/Quake-III-Arena.

The modifications in this project are intended to support building with VS 2010; however, I never completed it. The last working version I had was building in VS 2008. You'll need to sort out the compilation and project files yourself to get this working again.

All of the files here are intended to be drop-in replacements for the corresponding files in the Quake III repository. 

For a detailed overview of what this is intending to accomplish, please see my thesis: https://github.com/TylerGoeringer/Thesis or https://etd.ohiolink.edu/!etd.send_file?accession=case1373073319&disposition=inline.

Here's a brief description: these files will modify the Quake III source code to allow deathmatches using LSPI or Gradient learning bots. These bots will adapt over time to strategies, forming new techniques. These learning algorithms are applied to bots at a high level, affecting the goal states they seek, rather than determining individual actions (such as aiming, movement, and viewing). This means that the Quake III AI code is still in charge of how accurate someone is, how aggressive they are, etc. The concept here was that the fine grained mechanics were critical to imbue bots with personality as well as to allow different difficulty levels to matter. That is to say, a low difficulty bot may adapt it's strategy making it more difficult than typical, but it will still be very inaccurate.

The reward function can be used to tweak bot behavior and is found in ai_dmnet.h (you'll find that most of the important changes are here) in the function calculateReward().

The initialization of bots happens in ai_dmnet.h in the function AI_Init(). You can see in here a call to LspiBot_Init() regardless of Gradient or LSPI bot. This is because all bots are defined through the LspiBot interface. The code be_ai_lspi.cu handles differentiation of the internals on its own. If you wish to create a new type of bot, you should do so there and update the defines such as #define GRADIENT. For examples look at LspiAgent.h and GradientAgent.h.

Here's a brief breakdown of the #defines that you need to understand:

1) #define ONLINE_UPDATE (ai_dmnet.h) - This specifies whether the LSPI is using online or offline update. When defined the agent will update periodically while a match is ongoing.

2) #define MAX_UPDATES (ai_dmnet.h) - When ONLINE_UPDATE is defined, this value determines how many samples to store between successive updates. When offline mode is used this is irrelevant, all samples are saved to a file.

3) #define UPDATE_POLICY (ai_dmnet.h) - When this is defined, the LSPI update function will be called. The code for this starts in ai_dmnet.c and assumes the samples file is named "samples.dat". This is only called when the bot is first initialized. Note that you can define both ONLINE_UPDATE and UPDATE_POLICY can be defined. The bot will start initially based on the data and then update based on live results. Note that when this is enabled there will be a significant pause at the start of a match as the code is processed. This is normal, feel free to put debug print statements to ensure output is still being processed. (Look for the BotAI_Print() function for an idea on how to do this.)

4) #define COLLECT_SAMPLES (ai_dmnet.h) - When this is defined all samples will be saved to a file. You can see this code in ai_dmnet.c. The players in the game are 0 indexed, so each player (a maximum of 2 supported) will generate a sample file in the format "samplesX.dat" where X is the player index.

5) #define CPU (be_ai_lspi.h) - If defined uses the CPU to process all AI actions instead of the GPU. This requires libblas.lib in the LSPI folder. You may need to build your own libblas based on your system configuration.

6) #define GRADIENT (be_ai_lspi.h) - If defined then the agent will be a Gradient Agent instead of an LSPI Agent. Look in GradientAgent.h for more information. You'll note that there is a lot of commented out code around different basis sizes. This was for exploring the impact a larger basis had. You can uncomment the code as needed if you wish to do similar tests, or you can write your own basis_function and it should be more or less drop in code.

7) #define ONLINE (be_ai_lspi.h) - Should be shared with ONLINE_UPDATE. It allows the update code to execute on a different thread while the application is running.

8) #define EXPLORE (be_ai_lspi.h) - This only applies to the LSPI Agent, not the Gradient Agent. If the LSPI Agent is in use and this is defined the agent will choose actions randomly at a rate determined by EXP_RATE.

9) #define EXP_RATE (be_ai_lspi.h) - Determines that chance that an action will be selected randomly when EXPLORE is defined.

10) #define USER_SPECTATOR (g_local.h) - When defined, the first player is a spectator. In this case you should spawn 2 bots or else the bot will have no enemy. The configuration is set so that the first bot to spawn will always use the LSPI specified AI (which may be a Gradient bot instead of LSPI, depending on your other defines). The second bot will always use the game's AI although you should be able to change this if you choose to.

Hopefully this is sufficient to get started. I believe the most challenging part is simply getting it to build! Google around, there are a lot of people who have struggled with this. One particular thing I recommend: try building as different build types. I believe I only ever got a specific Debug build to function. And make sure you have the correct startup project (cgame or game if I remember correctly).

Good luck, and feel free to contact me if you have questions.
