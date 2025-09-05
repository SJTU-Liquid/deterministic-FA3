#pragma once

#include "cutlass/fast_math.h"
#include <cuda/std/iterator>


/*
Scheduler classes for the execution order of q tiles (m_block)
*/

namespace flash {

    // just the same as the original code, m grows from m_min to m_max
    class AscendingScheduler {
    private:
        int m_current;
        const int m_end;

    public:
        using iterator_category = cuda::std::forward_iterator_tag;
        using value_type        = int;
        using difference_type   = cuda::std::ptrdiff_t;
        using pointer           = const int*;
        using reference         = const int&;

        CUTLASS_DEVICE AscendingScheduler(int m_min, int m_max)
            : m_current(m_min), m_end(m_max) {}

        CUTLASS_DEVICE bool valid() const {
            return m_current < m_end;
        }

        CUTLASS_DEVICE AscendingScheduler& operator++() {
            ++m_current;
            return *this;
        }

        CUTLASS_DEVICE value_type operator*() const {
            return m_current;
        }
    };

    class DescendingScheduler {
    private:
        int m_current;
        const int m_end;
    public:
        using iterator_category = cuda::std::forward_iterator_tag;
        using value_type        = int;
        using difference_type   = cuda::std::ptrdiff_t;
        using pointer           = const int*;
        using reference         = const int&;

        CUTLASS_DEVICE DescendingScheduler(int m_min, int m_max)
            : m_current(m_max - 1), m_end(m_min - 1) {}

        CUTLASS_DEVICE bool valid() const {
            return m_current > m_end;
        }

        CUTLASS_DEVICE DescendingScheduler& operator++() {
            --m_current;
            return *this;
        }

        CUTLASS_DEVICE value_type operator*() const {
            return m_current;
        }
    };

    // targeting at full, so m_min must be 0, and m_max must be the same across all kv tiles
    // should be used with ShiftDependency
    // e.g. q_tile: 6 active_kv_tile: 3, schedule:
    // | 0    | 4    | 2    |
    // | 1    | 0    | 3    |
    // | 2    | 1    | 0    |
    // | 3    | 2    | 1    |
    // | 4    | 3    | 2    |
    class ShiftScheduler {
    public:
        int cur_step;
        const int start_q_idx;
        const int total_steps;
        
        using iterator_category = cuda::std::forward_iterator_tag;
        using value_type        = int;
        using difference_type   = cuda::std::ptrdiff_t;
        using pointer           = const int*;
        using reference         = const int&;

        CUTLASS_DEVICE ShiftScheduler(int m_min, int m_max, int active_kv_idx)
            : start_q_idx(active_kv_idx), cur_step(0), total_steps(m_max) {
            assert(m_min == 0); // must start from 0
            assert(num_inflight_kv_tiles <= m_max); // we assume there are more q tiles than kv tiles, otherwise conflict-free is impossible
        }

        CUTLASS_DEVICE bool valid() {
            return cur_step < total_steps;
        }

        CUTLASS_DEVICE ShiftScheduler& operator++() {
            ++cur_step;
            return *this;
        }

        CUTLASS_DEVICE value_type operator*() const {
            return (start_q_idx + cur_step) % total_steps; // wrap around
        }
    };

    class ShiftCausalScheduler {
    public:
        using iterator_category = cuda::std::forward_iterator_tag;
        using value_type        = int;
        using difference_type   = cuda::std::ptrdiff_t;
        using pointer           = const int*;
        using reference         = const int&;

        const int stage;
        int cur_step;
        const int total_steps;
        const int m_max;
        const int rectangle_start;
        const int sm_id;
        const int m_min;
        CUTLASS_DEVICE ShiftCausalScheduler(int m_min, int m_max, int active_sms, int sm_id, int stage, int stride, int global_m_min)
        : stage(stage), total_steps(m_max - m_min), cur_step(0), m_max(m_max), rectangle_start(global_m_min + stride * (active_sms - 1)), sm_id(sm_id), m_min(m_min) {}

        CUTLASS_DEVICE bool valid() {
            return cur_step < total_steps;
        }

        CUTLASS_DEVICE ShiftCausalScheduler& operator++() {
            ++cur_step;
            return *this;
        }

        CUTLASS_DEVICE value_type operator*() const {
            if (stage == 1) {
                // backward order
                return m_max - cur_step - 1;
            } else {
                // rotation in rectangle, forward in triangle
                int rectangle_steps = m_max - rectangle_start;
                if (cur_step < rectangle_steps) {
                    return (sm_id + cur_step) % rectangle_steps + rectangle_start;
                } else {
                    return m_min + (cur_step - rectangle_steps);
                }
            }
        }
    };
}