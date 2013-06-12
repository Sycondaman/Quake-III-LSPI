#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <cstdlib>
#include "math_constants.h"

// Force enums
#define NF_OPT 1
#define LF_OPT 2
#define RF_OPT 3
#define NF_FORCE 0
#define LF_FORCE -50
#define RF_FORCE 50

// Nonlinear equation constants
#define g_const 9.8f
#define m_const 2.0f
#define M_const 8.0f
#define l_const 0.5f
#define a_const 1.0f/(m_const+M_const)
#define noise 0
#define epsilon_const 0.00001f