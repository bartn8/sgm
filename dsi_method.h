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

#ifndef DSI_METHOD_H_
#define DSI_METHOD_H_

#include <stdint.h>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "debug.h"

void init_dsi_method();
void compute_dsi_method(uint8_t *h_dsi,  uint8_t *left_ct, uint8_t *right_ct, uint8_t *left, uint8_t *right, uint32_t h, uint32_t w, float *elapsed_time_ms);
void finish_dsi_method();

#endif /* DSI_METHOD_H_ */
