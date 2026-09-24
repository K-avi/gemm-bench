#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include "MicroKernelAdapter.h"
#include "StorageType.h"
#include "UArchMacroTraits.h"

namespace GEMMBench {

/**
 * @brief MacroKernel: Configurable, templated GotoBLAS/BLIS-style 5-loop GEMM.
 *
 * Implements: C = alpha * A * B + beta * C
 * 
 * @tparam T Floating-point datatype (float or double)
 * @tparam MC Number of rows in A panel (resident in L2 cache)
 * @tparam NC Number of columns in B panel (resident in L3 or shared cluster L2)
 * @tparam KC Shared accumulation dimension per tile
 * @tparam MicroKernelT Policy satisfying MicroKernelPolicy<T>
 * @tparam LayoutA Storage order of input matrix A (Row or Col)
 * @tparam LayoutB Storage order of input matrix B (Row or Col)
 */
template <
    typename T,
    size_t MC,
    size_t NC,
    size_t KC,
    typename MicroKernelT,
    PLayout LayoutA = PLayout::Row,
    PLayout LayoutB = PLayout::Row
>
requires MicroKernelPolicy<MicroKernelT, T>
class MacroKernel {
public:
    static constexpr size_t MR = MicroKernelT::MR;
    static constexpr size_t NR = MicroKernelT::NR;

    static_assert(MC % MR == 0, "MC must be an integer multiple of MR");
    static_assert(NC % NR == 0, "NC must be an integer multiple of NR");

    /**
     * @brief Executes full GEMM computation over arbitrary matrix dimensions M, N, K.
     * 
     * @param M Number of rows of A and C
     * @param N Number of columns of B and C
     * @param K Number of columns of A and rows of B
     * @param alpha Scalar multiplier for A * B
     * @param A Pointer to matrix A buffer
     * @param lda Leading dimension / stride of A
     * @param B Pointer to matrix B buffer
     * @param ldb Leading dimension / stride of B
     * @param beta Scalar multiplier for matrix C
     * @param C Pointer to output matrix C buffer
     * @param ldc Leading dimension / stride of C
     */
    static void gemm(size_t M, size_t N, size_t K,
                     T alpha,
                     const T* __restrict A, size_t lda,
                     const T* __restrict B, size_t ldb,
                     T beta,
                     T* __restrict C, size_t ldc) {
        // =====================================================================
        // STUB: MacroKernel 5-Loop Nest to be implemented in pair-programming
        // =====================================================================

        // Pack buffers allocated once per GEMM call (aligned to 64 bytes)
        // A_pack: MC x KC (MR-strip row-major blocks)
        // B_pack: KC x NC (NR-strip row-major blocks)
        alignas(64) static thread_local T a_pack[MC * KC];
        alignas(64) static thread_local T b_pack[KC * NC];

        // If beta is neither 0 nor 1, scale C by beta up-front (O(M*N) memory pass).
        // This allows all compute loops to operate purely in either overwrite (beta=0) or accumulate (beta=1) mode.
        if (beta != T{0} && beta != T{1}) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C[i * ldc + j] *= beta;
                }
            }
        }

        // Loop 5 (JC): over N in steps of NC
        for (size_t jc = 0; jc < N; jc += NC) {
            size_t nc_cur = std::min(NC, N - jc);

            // Loop 4 (PC): over K in steps of KC
            for (size_t pc = 0; pc < K; pc += KC) {
                size_t kc_cur = std::min(KC, K - pc);

                // Overwrite mode on first pc iteration if beta was originally 0.
                // Otherwise (pc > 0 or beta != 0), accumulate into C.
                const bool is_overwrite = (pc == 0 && beta == T{0});

                // STUB: Pack B panel: B(pc:pc+kc_cur, jc:jc+nc_cur) -> b_pack
                pack_B(B, ldb, pc, kc_cur, jc, nc_cur, b_pack);

                // Loop 3 (IC): over M in steps of MC
                for (size_t ic = 0; ic < M; ic += MC) {
                    size_t mc_cur = std::min(MC, M - ic);

                    // STUB: Pack A panel: A(ic:ic+mc_cur, pc:pc+kc_cur) -> a_pack (folds alpha)
                    pack_A(A, lda, ic, mc_cur, pc, kc_cur, a_pack, alpha);

                    // Loop 2 (JR): over NC in steps of NR
                    for (size_t jr = 0; jr < nc_cur; jr += NR) {
                        size_t nr_cur = std::min(NR, nc_cur - jr);

                        // Loop 1 (IR): over MC in steps of MR
                        for (size_t ir = 0; ir < mc_cur; ir += MR) {
                            size_t mr_cur = std::min(MR, mc_cur - ir);

                            const T* a_tile = a_pack + ir * KC;
                            const T* b_tile = b_pack + jr * KC;
                            T* c_tile = C + (ic + ir) * ldc + (jc + jr);

                            if (mr_cur == MR && nr_cur == NR) {
                                // Full register tile: fast path
                                if (is_overwrite) {
                                    MicroKernelT::compute_overwrite(a_tile, b_tile, c_tile, kc_cur, ldc);
                                } else {
                                    MicroKernelT::compute_accumulate(a_tile, b_tile, c_tile, kc_cur, ldc);
                                }
                            } else {
                                // STUB: Edge tile handling (zero-padded execution + selective writeback)
                                writeback_edge_tile(a_tile, b_tile, c_tile, ldc, mr_cur, nr_cur, kc_cur, is_overwrite);
                            }
                        }
                    }
                }
            }
        }
    }

private:
    /**
     * @brief Packs a sub-block of A of size mc_cur x kc_cur into contiguous A_pack.
     * 
     * Output layout: MC/MR strips of MR x KC row-major memory.
     * Zero-pads rows if mc_cur < MC or columns if kc_cur < KC.
     */
    static void pack_A(const T* __restrict A, size_t lda,
                       size_t ic, size_t mc_cur,
                       size_t pc, size_t kc_cur,
                       T* __restrict A_pack, T alpha = T{1}) {
        // =====================================================================
        // STUB: Implementation to be written together
        // =====================================================================
        (void)A; (void)lda; (void)ic; (void)mc_cur; (void)pc; (void)kc_cur; (void)A_pack; (void)alpha;
    }

    /**
     * @brief Packs a sub-block of B of size kc_cur x nc_cur into contiguous B_pack.
     * 
     * Output layout: NC/NR strips of KC x NR row-major memory.
     * Zero-pads columns if nc_cur < NC or rows if kc_cur < KC.
     */
    static void pack_B(const T* __restrict B, size_t ldb,
                       size_t pc, size_t kc_cur,
                       size_t jc, size_t nc_cur,
                       T* __restrict B_pack) {
        // =====================================================================
        // STUB: Implementation to be written together
        // =====================================================================
        (void)B; (void)ldb; (void)pc; (void)kc_cur; (void)jc; (void)nc_cur; (void)B_pack;
    }

    /**
     * @brief Edge-tile handler: computes full MR x NR into temporary buffer and writes back valid region.
     */
    static void writeback_edge_tile(const T* __restrict a_tile,
                                    const T* __restrict b_tile,
                                    T* __restrict c_dst, size_t ldc,
                                    size_t mr_cur, size_t nr_cur, size_t kc,
                                    bool is_overwrite) {
        // =====================================================================
        // STUB: Implementation to be written together
        // =====================================================================
        (void)a_tile; (void)b_tile; (void)c_dst; (void)ldc;
        (void)mr_cur; (void)nr_cur; (void)kc; (void)is_overwrite;
    }
};

/**
 * @brief Helper alias to instantiate a MacroKernel from a TargetUArch and microkernel policy.
 */
template <TargetUArch Arch, typename T, typename MicroKernelT = UArchChampionPolicy_t<Arch, T>>
using UArchMacroKernel = MacroKernel<
    T,
    UArchMacroTraits<Arch>::MC,
    UArchMacroTraits<Arch>::NC,
    UArchMacroTraits<Arch>::KC,
    MicroKernelT
>;

} // namespace GEMMBench
