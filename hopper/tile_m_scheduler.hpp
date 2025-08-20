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

        CUTLASS_DEVICE AscendingScheduler& begin() {
            return *this;
        }

        CUTLASS_DEVICE AscendingScheduler end() {
            return AscendingScheduler(m_end, m_end);
        }

        CUTLASS_DEVICE bool operator!=(const AscendingScheduler& other) const {
            return this->m_current < other.m_current;
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
        CUTLASS_DEVICE DescendingScheduler(int end_val) 
            : m_current(end_val), m_end(end_val) {}

    public:
        using iterator_category = cuda::std::forward_iterator_tag;
        using value_type        = int;
        using difference_type   = cuda::std::ptrdiff_t;
        using pointer           = const int*;
        using reference         = const int&;

        CUTLASS_DEVICE DescendingScheduler(int m_min, int m_max)
            : m_current(m_max - 1), m_end(m_min - 1) {}

        CUTLASS_DEVICE DescendingScheduler& begin() {
            return *this;
        }

        CUTLASS_DEVICE DescendingScheduler end() {
            return DescendingScheduler(m_end);
        }

        CUTLASS_DEVICE bool operator!=(const DescendingScheduler& other) const {
            return this->m_current > other.m_end;
        }

        CUTLASS_DEVICE DescendingScheduler& operator++() {
            --m_current;
            return *this;
        }

        CUTLASS_DEVICE value_type operator*() const {
            return m_current;
        }
    };

    
}