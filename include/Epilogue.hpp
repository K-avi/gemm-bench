#pragma once

#include "mipp.hpp"

namespace GEMMBench {

struct Epilogue {
  //
  // Épilogue scalaire
  //
  template <typename T, bool FastPath = false>
  static inline void store(T *__restrict ptr, T acc, T alpha, T beta) {
    if constexpr (FastPath) {
      *ptr = acc;
    } else {
      if (beta == T{0}) {
        *ptr = (alpha == T{1}) ? acc : (alpha * acc);
      } else if (beta == T{1}) {
        *ptr = (alpha == T{1}) ? (*ptr + acc) : (*ptr + alpha * acc);
      } else {
        *ptr = (alpha == T{1}) ? (beta * (*ptr) + acc)
                               : (beta * (*ptr) + alpha * acc);
      }
    }
  }

  //
  // Épilogue vectoriel SIMD / RVV (MIPP v2)
  //
  template <typename T, int lmul = 1, bool FastPath = false, typename VReg>
  static inline void store(T *__restrict ptr, const VReg &acc, T alpha, T beta) {
    using namespace mipp;

    if constexpr (FastPath) {
      mipp::store(ptr, acc);
    } else {
      if (beta == T{0}) {
        if (alpha == T{1}) {
          mipp::store(ptr, acc);
        } else {
          mipp::store(ptr, mul(acc, set1<T, lmul>(alpha)));
        }
      } else if (beta == T{1}) {
        auto c_old = load<T, lmul>(ptr);
        if (alpha == T{1}) {
          mipp::store(ptr, add(c_old, acc));
        } else {
          mipp::store(ptr, fmadd(acc, set1<T, lmul>(alpha), c_old));
        }
      } else {
        auto c_old = load<T, lmul>(ptr);
        auto c_scaled = mul(c_old, set1<T, lmul>(beta));
        if (alpha == T{1}) {
          mipp::store(ptr, add(c_scaled, acc));
        } else {
          mipp::store(ptr, fmadd(acc, set1<T, lmul>(alpha), c_scaled));
        }
      }
    }
  }
};

} // namespace GEMMBench
