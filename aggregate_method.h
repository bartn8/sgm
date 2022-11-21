/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2022 Luca Bartolomei.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef AGGREGATE_METHOD_H_
#define AGGREGATE_METHOD_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "cost_aggregation.h"
#include "debug.h"

typedef struct aggregate_tuple_struct
{
    uint16_t *idsi;
    uint8_t *disp;
} aggregate_tuple;

#if PATH_AGGREGATION == 8
__global__ void addKernel(uint16_t* c, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const uint8_t* d_L7, int size);
#else
__global__ void addKernel(uint16_t* c, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, int size);
#endif

void init_aggregate_method(const uint8_t _p1, const uint8_t _p2);
aggregate_tuple compute_aggregate_method(cost_t *left_ct, cost_t *right_ct, uint32_t h, uint32_t w, uint8_t *h_dsi, float *elapsed_time_ms);
void finish_aggregate_method();
static void free_aggregate_memory();

#endif /* AGGREGATE_METHOD_H_ */
