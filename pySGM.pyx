# cython: language_level=3

from cpython cimport array
import array

import numpy as np
cimport numpy as np

from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)


np.import_array()

cdef extern from "configuration.h":
    ctypedef uint32_t cost_t
    cdef int MAX_DISPARITY

cdef extern from "dsi_method.h":
    void init_dsi_method()
    void compute_dsi_method(uint8_t *h_dsi, uint8_t *left_ct, uint8_t *right_ct, uint8_t *left, uint8_t *right, uint32_t h, uint32_t w, float *elapsed_time_ms)
    void finish_dsi_method()

cdef extern from "aggregate_method.h":
    void init_aggregate_method(const uint8_t _p1, const uint8_t _p2)
    void compute_aggregate_method(uint16_t *h_idsi, uint8_t *h_disparity, cost_t *left_ct, cost_t *right_ct, uint8_t *h_dsi, uint32_t h, uint32_t w, float *elapsed_time_ms)
    void finish_aggregate_method()

def init_sgm():
    init_dsi_method()
    init_aggregate_method()

def dispose_sgm():
    finish_dsi_method()
    finish_aggregate_method()

def calculateDSI(left, right):
    cdef uint32_t h,w
    cdef np.ndarray[np.uint8_t, ndim=1, mode = 'c'] dsi, flat_left, flat_right
    cdef uint8_t *dsi_ptr, *flat_left_ptr, *flat_right_ptr
    cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] flat_left_ct, flat_right_ct
    cdef cost_t *flat_left_ct_ptr, *flat_right_ct_ptr
    cdef float elapsed_time_ms
    
    h,w = left.shape[:2]
    
    flat_left = np.ascontiguousarray(left.flatten(), dtype=np.uint8_t)
    flat_left_ptr = <uint8_t*>flat_left.data

    flat_right = np.ascontiguousarray(right.flatten(), dtype=np.uint8_t)
    flat_right_ptr = <uint8_t*>flat_right.data
    
    flat_left_ct = np.ascontiguousarray(np.zeros((h*w), dtype=np.uint32_t))
    flat_left_ct_ptr = <cost_t*>flat_left_ct.data

    flat_right_ct = np.ascontiguousarray(np.zeros((h*w), dtype=np.uint32_t))
    flat_right_ct_ptr = <cost_t*>flat_right_ct.data

    dsi = np.ascontiguousarray(np.zeros((h*w*MAX_DISPARITY), dtype=np.uint8_t))
    dsi_ptr = <uint8_t*>dsi.data

    compute_dsi_method(dsi_ptr, flat_left_ct_ptr, flat_right_ct_ptr, flat_left_ptr, flat_right_ptr, h, w, &elapsed_time_ms)

    return np.reshape(dsi, (h,w,MAX_DISPARITY)), np.reshape(flat_left_ct_ptr, (h,w)), np.reshape(flat_right_ct_ptr, (h,w)),  elapsed_time_ms

def aggregateDSI(left_ct, right_ct, dsi):
    cdef uint32_t h,w
    cdef np.ndarray[np.uint16_t, ndim=1, mode = 'c'] idsi
    cdef uint16_t *idsi_ptr
    cdef np.ndarray[np.uint8_t, ndim=1, mode = 'c'] disparity, flat_dsi
    cdef uint8_t *disparity_ptr, *flat_dsi_ptr
    cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] flat_left_ct, flat_right_ct
    cdef cost_t *flat_left_ct_ptr, *flat_right_ct_ptr
    cdef float elapsed_time_ms
    
    h,w = left_ct.shape[:2]
    
    flat_left_ct = np.ascontiguousarray(left_ct.flatten(), dtype=np.uint32_t)
    flat_left_ct_ptr = <cost_t*>flat_left_ct.data

    flat_right_ct = np.ascontiguousarray(right_ct.flatten(), dtype=np.uint32_t)
    flat_right_ct_ptr = <cost_t*>flat_right_ct.data
    
    flat_dsi = np.ascontiguousarray(dsi.flatten(), dtype=np.uint8_t)
    flat_dsi_ptr = <uint8_t*>flat_dsi.data

    idsi = np.ascontiguousarray(np.zeros((h*w*MAX_DISPARITY), dtype=np.uint16_t))
    idsi_ptr = <uint16_t*>idsi.data

    disparity = np.ascontiguousarray(np.zeros((h*w), dtype=np.uint8_t))
    disparity_ptr = <uint8_t*>disparity.data

    compute_aggregate_method(idsi_ptr, disparity_ptr, flat_left_ct_ptr, flat_right_ct_ptr, flat_dsi_ptr, h, w, &elapsed_time_ms)

    return np.reshape(idsi, (h,w,MAX_DISPARITY)), np.reshape(disparity, (h,w)), elapsed_time_ms