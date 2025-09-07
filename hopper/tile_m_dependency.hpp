#pragma once

#include "cutlass/fast_math.h"
#include "tile_m_scheduler.hpp"


/*
functions for the dependency order of q tiles (m_block)
*/

namespace flash {

    // the original dependency order
    class NaiveDependency {
    public:
        CUTLASS_DEVICE int operator()(int kv_id) const {
            return kv_id;
        }
    };

    // Used with ShiftScheduler
    // e.g. q_tile: 6 active_kv_tile: 3, 
    // schedule:
    // | 0    | 4    | 2    |
    // | 1    | 0    | 3    |
    // | 2    | 1    | 0    |
    // | 3    | 2    | 1    |
    // | 4    | 3    | 2    |
    // dependency:
    // | 0    | 2    | 1    |
    // | 1    | 0    | 2    |
    // | 2    | 1    | 0    |
    // | 2    | 1    | 0    |
    // | 2    | 1    | 0    |
    class ShiftDependency {
    public:
        CUTLASS_DEVICE short operator()(short q_id, short active_kv_id, short active_kv_tiles, short executed_kv_tiles) const {
            short effective_row = min(q_id, active_kv_tiles - 1);
            short tmp = effective_row - active_kv_id;
            if (tmp < 0) tmp += active_kv_tiles;
            return executed_kv_tiles + tmp;
            // return executed_kv_tiles + (effective_row - active_kv_id + active_kv_tiles) % active_kv_tiles;
        }
    };

    class ShiftCausalDependency {
    public:
        CUTLASS_DEVICE short operator()(ShiftCausalScheduler const& scheduler, short active_sms, short executed_kvs, short stride) const {
            if (scheduler.stage == 0) {
                short rectangle_steps = scheduler.m_max - scheduler.rectangle_start;
                if (scheduler.cur_step < rectangle_steps) {
                    short tmp = (scheduler.sm_id + scheduler.cur_step);
                    if (tmp >= rectangle_steps) tmp -= rectangle_steps;
                    return ShiftDependency()(tmp, scheduler.sm_id, active_sms, executed_kvs);
                } else {
                    return (scheduler.cur_step - rectangle_steps) / stride + executed_kvs;
                }
            } else {
                return active_sms - 1 - scheduler.sm_id + executed_kvs + active_sms;
            }
        }
    };
}