#pragma once

#include <cstdint>

// Compiler memory barrier: prevents compiler instruction reordering across the barrier
inline void compiler_fence() {
    asm volatile("" ::: "memory");
}

// Hardware execution pipeline fence: serializes the CPU out-of-order execution engine
inline void pipeline_fence() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    asm volatile("lfence" ::: "memory");
#elif defined(__aarch64__)
    asm volatile("isb" ::: "memory");
#elif defined(__riscv)
    asm volatile("fence i,r" ::: "memory");
#else
    compiler_fence();
#endif
}
