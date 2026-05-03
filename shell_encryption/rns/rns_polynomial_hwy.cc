// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shell_encryption/rns/rns_polynomial_hwy.h"

#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/types/span.h"
#include "hwy/detect_targets.h"
#include "shell_encryption/integral_types.h"
#include "shell_encryption/montgomery.h"

// Highway implementations.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "shell_encryption/rns/rns_polynomial_hwy.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// clang-format on

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace rlwe::internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

#if HWY_TARGET == HWY_SCALAR

template <typename Integer>
void BatchFusedMulAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  BatchFusedMulAddMontgomeryRepNoHwy(a, b, output);
}

template <typename Integer>
void BatchAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  BatchAddMontgomeryRepNoHwy(a, b, output);
}

template <typename Integer>
void BatchSubMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b, Integer q,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  BatchSubMontgomeryRepNoHwy(a, b, output);
}

#else

// General version that should work for `Integer` smaller than 64-bit unsigned
// types.
template <typename Integer>
void BatchFusedMulAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  using D = hn::ScalableTag<Integer>;
  using Wide = hwy::MakeWide<hn::TFromD<D>>;
  const D d;                          // For lanes containing Integers.
  const hn::Repartition<Wide, D> d2;  // For double width products of Integers.

  const int N = hn::Lanes(d);  // This is guaranteed to be a power of two.
  const int64_t num_coeffs = a.size();
  int64_t i = 0;
  for (; i + N <= num_coeffs; i += N) {
    BigInteger* output0_ptr = &output[i];
    BigInteger* output1_ptr = &output[i] + N / 2;
    const Integer* a_ptr = reinterpret_cast<const Integer*>(&a[i]);
    const Integer* b_ptr = reinterpret_cast<const Integer*>(&b[i]);
    auto output0 = hn::Load(d2, output0_ptr);
    auto output1 = hn::Load(d2, output1_ptr);
    auto a_vec = hn::LoadU(d, a_ptr);  // a[i..i+N]
    auto b_vec = hn::LoadU(d, b_ptr);  // b[i..i+N]
    // Compute a[even_idx] * b[even_idx].
    auto mul_even = hn::MulEven(a_vec, b_vec);
    // Compute a[odd_idx] * b[odd_idx].
    auto mul_odd = hn::MulOdd(a_vec, b_vec);
    // Merge the lower blocks into mul0, and upper blocks into mul1.
    auto mul0 = hn::InterleaveWholeLower(d2, mul_even, mul_odd);
    auto mul1 = hn::InterleaveWholeUpper(d2, mul_even, mul_odd);
    // Add the lower and higher blocks of products to the output vector.
    hn::Store(hn::Add(output0, mul0), d2, output0_ptr);
    hn::Store(hn::Add(output1, mul1), d2, output1_ptr);
  }

  // For the remaining elements in `a` and `b`.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInteger>(a[i].GetMontgomeryRepresentation()) *
                 b[i].GetMontgomeryRepresentation();
  }
}

using ModularInt64 = MontgomeryInt<Uint64>;
using BigInt64 = ModularInt64::BigInt;

// Specialized version for 64-bit integers, as multiplication needs more careful
// treatment.
template <>
void BatchFusedMulAddMontgomeryRepHwy(absl::Span<const ModularInt64> a,
                                      absl::Span<const ModularInt64> b,
                                      hwy::AlignedVector<BigInt64>& output) {
  using D = hn::ScalableTag<Uint64>;
  const D d;
  const int N = hn::Lanes(d);
  const int64_t num_coeffs = a.size();

  // Generate the masks on the even lanes, which correspond to the lower 64 bits
  // of BigInt64 (unsigned 128-bit int) values in the output vector.
  uint8_t* mask_lo_bits = new uint8_t[(N + 7) / 8];
  for (int j = 0; j < (N + 7) / 8; ++j) {
    mask_lo_bits[j] = 0;
    for (int k = 0; k < 8; k += 2) {
      mask_lo_bits[j] |= static_cast<uint8_t>(1 << k);
    }
  }
  auto mask_lo = hn::LoadMaskBits(d, mask_lo_bits);
  // A highway vector whose odd lanes (higher 64 bits of the output vector) are
  // all 1's and even lanes are all 0's.
  auto ones = hn::Slide1Up(d, hn::IfThenElseZero(mask_lo, hn::Set(d, 1)));

  int64_t i = 0;

  for (; i + N <= num_coeffs; i += N) {
    // Load the lower N/2 and upper N/2 values from `output`.
    Uint64* output0_ptr = reinterpret_cast<Uint64*>(&output[i]);
    Uint64* output1_ptr = reinterpret_cast<Uint64*>(&output[i]) + N;
    auto output0 = hn::Load(d, output0_ptr);
    auto output1 = hn::Load(d, output1_ptr);

    const Uint64* a_ptr = reinterpret_cast<const Uint64*>(&a[i]);
    const Uint64* b_ptr = reinterpret_cast<const Uint64*>(&b[i]);
    auto a_vec = hn::LoadU(d, a_ptr);
    auto b_vec = hn::LoadU(d, b_ptr);

    // `hn::MulEven` and `hn::MulOdd` on 64-bit inputs produce a vector of
    // 64-bit values with even lanes storing the lower half of product and odd
    // lanes storing the upper half. That is,
    //   mul_even = [ab[0].lo, ab[0].hi, ab[2].lo, ab[2].hi, ...]
    //   mul_odd  = [ab[1].lo, ab[1].hi, ab[3].lo, ab[3].hi, ...]
    // So we first merge the lower and upper halves, and then permute within
    // each block of four 64-bit values, and get
    //   mul0 = [ab[0].lo, ab[0].hi, ab[1].lo, ab[1].hi, ...]
    //   mul1 = [ab[N/2].lo, ab[N/2].hi, ab[N/2+1].lo, ab[N/2+1].hi, ...]
    auto mul_even = hn::MulEven(a_vec, b_vec);
    auto mul_odd = hn::MulOdd(a_vec, b_vec);
    auto mul_interleave0 = hn::InterleaveWholeLower(d, mul_even, mul_odd);
    auto mul_interleave1 = hn::InterleaveWholeUpper(d, mul_even, mul_odd);
    auto mul0 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave0);
    auto mul1 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave1);

    // Add the products to the output lanes. A carry bit occurs when the new
    // lower 64-bit value becomes smaller, and we add 1 to the upper 64-bit lane
    // if that happens.
    auto output0_new = hn::Add(output0, mul0);
    auto output1_new = hn::Add(output1, mul1);
    auto mask_carry0 = hn::And(hn::Lt(output0_new, output0), mask_lo);
    auto mask_carry1 = hn::And(hn::Lt(output1_new, output1), mask_lo);
    mask_carry0 = hn::SlideMask1Up(d, mask_carry0);  // move the mask to hi64
    mask_carry1 = hn::SlideMask1Up(d, mask_carry1);  // move the mask to hi64
    output0 = hn::MaskedAddOr(output0_new, mask_carry0, output0_new, ones);
    output1 = hn::MaskedAddOr(output1_new, mask_carry1, output1_new, ones);

    hn::Store(output0, d, output0_ptr);
    hn::Store(output1, d, output1_ptr);
  }
  delete[] mask_lo_bits;

  // Handle the remaining elements in the input vectors.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInt64>(a[i].GetMontgomeryRepresentation()) *
                 b[i].GetMontgomeryRepresentation();
  }
}

////////////////////////////////////////////////////////////////////////////////
// FMSumAdd
////////////////////////////////////////////////////////////////////////////////

template <typename Integer>
void BatchFusedMulSumAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    absl::Span<const MontgomeryInt<Integer>> c,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  using D = hn::ScalableTag<Integer>;
  using Wide = hwy::MakeWide<hn::TFromD<D>>;
  const D d;                          // For lanes containing Integers.
  const hn::Repartition<Wide, D> d2;  // For double width products of Integers.

  const int N = hn::Lanes(d);  // This is guaranteed to be a power of two.
  const int64_t num_coeffs = a.size();
  int64_t i = 0;
  for (; i + N <= num_coeffs; i += N) {
    BigInteger* output0_ptr = &output[i];
    BigInteger* output1_ptr = &output[i] + N / 2;
    const Integer* a_ptr = reinterpret_cast<const Integer*>(&a[i]);
    const Integer* b_ptr = reinterpret_cast<const Integer*>(&b[i]);
    const Integer* c_ptr = reinterpret_cast<const Integer*>(&c[i]);
    auto output0 = hn::Load(d2, output0_ptr);
    auto output1 = hn::Load(d2, output1_ptr);
    auto a_vec = hn::LoadU(d, a_ptr);  // a[i..i+N]
    auto b_vec = hn::LoadU(d, b_ptr);  // b[i..i+N]
    auto c_vec = hn::LoadU(d, c_ptr);  // b[i..i+N]

    // Compute a[i] + b[i].
    auto ab_vec = hn::Add(a_vec, b_vec);
    // Compute ab[even_idx] * c[even_idx].
    auto mul_even = hn::MulEven(ab_vec, c_vec);
    // Compute ab[odd_idx] * c[odd_idx].
    auto mul_odd = hn::MulOdd(ab_vec, b_vec);

    // Merge the lower blocks into mul0, and upper blocks into mul1.
    auto mul0 = hn::InterleaveWholeLower(d2, mul_even, mul_odd);
    auto mul1 = hn::InterleaveWholeUpper(d2, mul_even, mul_odd);
    // Add the lower and higher blocks of products to the output vector.
    hn::Store(hn::Add(output0, mul0), d2, output0_ptr);
    hn::Store(hn::Add(output1, mul1), d2, output1_ptr);
  }

  // For the remaining elements in `a`, `b`, and `c`.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInteger>(a[i].GetMontgomeryRepresentation() +
                                         b[i].GetMontgomeryRepresentation()) *
                 c[i].GetMontgomeryRepresentation();
  }
}

using ModularInt64 = MontgomeryInt<Uint64>;
using BigInt64 = ModularInt64::BigInt;

// Specialized version for 64-bit integers, as multiplication needs more careful
// treatment.
template <>
void BatchFusedMulSumAddMontgomeryRepHwy(absl::Span<const ModularInt64> a,
                                         absl::Span<const ModularInt64> b,
                                         absl::Span<const ModularInt64> c,
                                         hwy::AlignedVector<BigInt64>& output) {
  using D = hn::ScalableTag<Uint64>;
  const D d;
  const int N = hn::Lanes(d);
  const int64_t num_coeffs = a.size();

  // Generate the masks on the even lanes, which correspond to the lower 64 bits
  // of BigInt64 (unsigned 128-bit int) values in the output vector.
  uint8_t* mask_lo_bits = new uint8_t[(N + 7) / 8];
  for (int j = 0; j < (N + 7) / 8; ++j) {
    mask_lo_bits[j] = 0;
    for (int k = 0; k < 8; k += 2) {
      mask_lo_bits[j] |= static_cast<uint8_t>(1 << k);
    }
  }
  auto mask_lo = hn::LoadMaskBits(d, mask_lo_bits);
  // A highway vector whose odd lanes (higher 64 bits of the output vector) are
  // all 1's and even lanes are all 0's.
  auto ones = hn::Slide1Up(d, hn::IfThenElseZero(mask_lo, hn::Set(d, 1)));

  int64_t i = 0;

  for (; i + N <= num_coeffs; i += N) {
    // Load the lower N/2 and upper N/2 values from `output`.
    Uint64* output0_ptr = reinterpret_cast<Uint64*>(&output[i]);
    Uint64* output1_ptr = reinterpret_cast<Uint64*>(&output[i]) + N;
    auto output0 = hn::Load(d, output0_ptr);
    auto output1 = hn::Load(d, output1_ptr);

    const Uint64* a_ptr = reinterpret_cast<const Uint64*>(&a[i]);
    const Uint64* b_ptr = reinterpret_cast<const Uint64*>(&b[i]);
    const Uint64* c_ptr = reinterpret_cast<const Uint64*>(&c[i]);
    auto a_vec = hn::LoadU(d, a_ptr);
    auto b_vec = hn::LoadU(d, b_ptr);
    auto c_vec = hn::LoadU(d, c_ptr);

    // Compute a[i] + b[i], assuming no overflow.
    auto ab_vec = hn::Add(a_vec, b_vec);

    // `hn::MulEven` and `hn::MulOdd` on 64-bit inputs produce a vector of
    // 64-bit values with even lanes storing the lower half of product and odd
    // lanes storing the upper half. That is,
    //   mul_even = [abc[0].lo, abc[0].hi, abc[2].lo, abc[2].hi, ...]
    //   mul_odd  = [abc[1].lo, abc[1].hi, abc[3].lo, abc[3].hi, ...]
    // So we first merge the lower and upper halves, and then permute within
    // each block of four 64-bit values, and get
    //   mul0 = [abc[0].lo, abc[0].hi, abc[1].lo, abc[1].hi, ...]
    //   mul1 = [abc[N/2].lo, abc[N/2].hi, abc[N/2+1].lo, abc[N/2+1].hi, ...]
    auto mul_even = hn::MulEven(ab_vec, c_vec);
    auto mul_odd = hn::MulOdd(ab_vec, c_vec);
    auto mul_interleave0 = hn::InterleaveWholeLower(d, mul_even, mul_odd);
    auto mul_interleave1 = hn::InterleaveWholeUpper(d, mul_even, mul_odd);
    auto mul0 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave0);
    auto mul1 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave1);

    // Add the products to the output lanes. A carry bit occurs when the new
    // lower 64-bit value becomes smaller, and we add 1 to the upper 64-bit lane
    // if that happens.
    auto output0_new = hn::Add(output0, mul0);
    auto output1_new = hn::Add(output1, mul1);
    auto mask_carry0 = hn::And(hn::Lt(output0_new, output0), mask_lo);
    auto mask_carry1 = hn::And(hn::Lt(output1_new, output1), mask_lo);
    mask_carry0 = hn::SlideMask1Up(d, mask_carry0);  // move the mask to hi64
    mask_carry1 = hn::SlideMask1Up(d, mask_carry1);  // move the mask to hi64
    output0 = hn::MaskedAddOr(output0_new, mask_carry0, output0_new, ones);
    output1 = hn::MaskedAddOr(output1_new, mask_carry1, output1_new, ones);

    hn::Store(output0, d, output0_ptr);
    hn::Store(output1, d, output1_ptr);
  }
  delete[] mask_lo_bits;

  // Handle the remaining elements in the input vectors.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInt64>(a[i].GetMontgomeryRepresentation() +
                                       b[i].GetMontgomeryRepresentation()) *
                 c[i].GetMontgomeryRepresentation();
  }
}

////////////////////////////////////////////////////////////////////////////////
// FMDifferenceAdd
////////////////////////////////////////////////////////////////////////////////

template <typename Integer>
void BatchFusedMulDifferenceAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    absl::Span<const MontgomeryInt<Integer>> c, Integer q,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  using D = hn::ScalableTag<Integer>;
  using Wide = hwy::MakeWide<hn::TFromD<D>>;
  const D d;                          // For lanes containing Integers.
  const hn::Repartition<Wide, D> d2;  // For double width products of Integers.

  const int N = hn::Lanes(d);  // This is guaranteed to be a power of two.
  const int64_t num_coeffs = a.size();
  auto q_vec = hn::Set(d, q);
  int64_t i = 0;
  for (; i + N <= num_coeffs; i += N) {
    BigInteger* output0_ptr = &output[i];
    BigInteger* output1_ptr = &output[i] + N / 2;
    const Integer* a_ptr = reinterpret_cast<const Integer*>(&a[i]);
    const Integer* b_ptr = reinterpret_cast<const Integer*>(&b[i]);
    const Integer* c_ptr = reinterpret_cast<const Integer*>(&c[i]);
    auto output0 = hn::Load(d2, output0_ptr);
    auto output1 = hn::Load(d2, output1_ptr);
    auto a_vec = hn::LoadU(d, a_ptr);  // a[i..i+N]
    auto b_vec = hn::LoadU(d, b_ptr);  // b[i..i+N]
    auto c_vec = hn::LoadU(d, c_ptr);  // b[i..i+N]

    // Compute a[i] - b[i] (mod q).
    auto underflow_mask = hn::Lt(a_vec, b_vec);
    auto aa_vec = hn::IfThenElse(underflow_mask, hn::Add(a_vec, q_vec), a_vec);
    auto ab_vec = hn::Sub(aa_vec, b_vec);

    // Compute ab[even_idx] * c[even_idx].
    auto mul_even = hn::MulEven(ab_vec, c_vec);
    // Compute ab[odd_idx] * c[odd_idx].
    auto mul_odd = hn::MulOdd(ab_vec, b_vec);

    // Merge the lower blocks into mul0, and upper blocks into mul1.
    auto mul0 = hn::InterleaveWholeLower(d2, mul_even, mul_odd);
    auto mul1 = hn::InterleaveWholeUpper(d2, mul_even, mul_odd);
    // Add the lower and higher blocks of products to the output vector.
    hn::Store(hn::Add(output0, mul0), d2, output0_ptr);
    hn::Store(hn::Add(output1, mul1), d2, output1_ptr);
  }

  // For the remaining elements in `a`, `b`, and `c`.
  for (; i < num_coeffs; ++i) {
    output[i] +=
        static_cast<BigInteger>(a[i].GetMontgomeryRepresentation() + q -
                                b[i].GetMontgomeryRepresentation()) *
        c[i].GetMontgomeryRepresentation();
  }
}

using ModularInt64 = MontgomeryInt<Uint64>;
using BigInt64 = ModularInt64::BigInt;

// Specialized version for 64-bit integers, as multiplication needs more careful
// treatment.
template <>
void BatchFusedMulDifferenceAddMontgomeryRepHwy(
    absl::Span<const ModularInt64> a, absl::Span<const ModularInt64> b,
    absl::Span<const ModularInt64> c, Uint64 q,
    hwy::AlignedVector<BigInt64>& output) {
  using D = hn::ScalableTag<Uint64>;
  const D d;
  const int N = hn::Lanes(d);
  const int64_t num_coeffs = a.size();

  // Generate the masks on the even lanes, which correspond to the lower 64 bits
  // of BigInt64 (unsigned 128-bit int) values in the output vector.
  uint8_t* mask_lo_bits = new uint8_t[(N + 7) / 8];
  for (int j = 0; j < (N + 7) / 8; ++j) {
    mask_lo_bits[j] = 0;
    for (int k = 0; k < 8; k += 2) {
      mask_lo_bits[j] |= static_cast<uint8_t>(1 << k);
    }
  }
  auto mask_lo = hn::LoadMaskBits(d, mask_lo_bits);
  // A highway vector whose odd lanes (higher 64 bits of the output vector) are
  // all 1's and even lanes are all 0's.
  auto ones = hn::Slide1Up(d, hn::IfThenElseZero(mask_lo, hn::Set(d, 1)));
  auto q_vec = hn::Set(d, q);

  int64_t i = 0;

  for (; i + N <= num_coeffs; i += N) {
    // Load the lower N/2 and upper N/2 values from `output`.
    Uint64* output0_ptr = reinterpret_cast<Uint64*>(&output[i]);
    Uint64* output1_ptr = reinterpret_cast<Uint64*>(&output[i]) + N;
    auto output0 = hn::Load(d, output0_ptr);
    auto output1 = hn::Load(d, output1_ptr);

    const Uint64* a_ptr = reinterpret_cast<const Uint64*>(&a[i]);
    const Uint64* b_ptr = reinterpret_cast<const Uint64*>(&b[i]);
    const Uint64* c_ptr = reinterpret_cast<const Uint64*>(&c[i]);
    auto a_vec = hn::LoadU(d, a_ptr);
    auto b_vec = hn::LoadU(d, b_ptr);
    auto c_vec = hn::LoadU(d, c_ptr);

    // Compute a[i] - b[i] (mod q).
    auto underflow_mask = hn::Lt(a_vec, b_vec);
    auto aa_vec = hn::IfThenElse(underflow_mask, hn::Add(a_vec, q_vec), a_vec);
    auto ab_vec = hn::Sub(aa_vec, b_vec);

    // `hn::MulEven` and `hn::MulOdd` on 64-bit inputs produce a vector of
    // 64-bit values with even lanes storing the lower half of product and odd
    // lanes storing the upper half. That is,
    //   mul_even = [abc[0].lo, abc[0].hi, abc[2].lo, abc[2].hi, ...]
    //   mul_odd  = [abc[1].lo, abc[1].hi, abc[3].lo, abc[3].hi, ...]
    // So we first merge the lower and upper halves, and then permute within
    // each block of four 64-bit values, and get
    //   mul0 = [abc[0].lo, abc[0].hi, abc[1].lo, abc[1].hi, ...]
    //   mul1 = [abc[N/2].lo, abc[N/2].hi, abc[N/2+1].lo, abc[N/2+1].hi, ...]
    auto mul_even = hn::MulEven(ab_vec, c_vec);
    auto mul_odd = hn::MulOdd(ab_vec, c_vec);
    auto mul_interleave0 = hn::InterleaveWholeLower(d, mul_even, mul_odd);
    auto mul_interleave1 = hn::InterleaveWholeUpper(d, mul_even, mul_odd);
    auto mul0 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave0);
    auto mul1 = hn::Per4LaneBlockShuffle<3, 1, 2, 0>(mul_interleave1);

    // Add the products to the output lanes. A carry bit occurs when the new
    // lower 64-bit value becomes smaller, and we add 1 to the upper 64-bit lane
    // if that happens.
    auto output0_new = hn::Add(output0, mul0);
    auto output1_new = hn::Add(output1, mul1);
    auto mask_carry0 = hn::And(hn::Lt(output0_new, output0), mask_lo);
    auto mask_carry1 = hn::And(hn::Lt(output1_new, output1), mask_lo);
    mask_carry0 = hn::SlideMask1Up(d, mask_carry0);  // move the mask to hi64
    mask_carry1 = hn::SlideMask1Up(d, mask_carry1);  // move the mask to hi64
    output0 = hn::MaskedAddOr(output0_new, mask_carry0, output0_new, ones);
    output1 = hn::MaskedAddOr(output1_new, mask_carry1, output1_new, ones);

    hn::Store(output0, d, output0_ptr);
    hn::Store(output1, d, output1_ptr);
  }
  delete[] mask_lo_bits;

  // Handle the remaining elements in the input vectors.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInt64>(a[i].GetMontgomeryRepresentation() +
                                       b[i].GetMontgomeryRepresentation()) *
                 c[i].GetMontgomeryRepresentation();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Add
////////////////////////////////////////////////////////////////////////////////

template <typename Integer>
void BatchAddMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  using D = hn::ScalableTag<Integer>;
  using Wide = hwy::MakeWide<hn::TFromD<D>>;
  const D d;                          // For lanes containing Integers.
  const hn::Repartition<Wide, D> d2;  // For double width products of Integers.

  const int N = hn::Lanes(d);  // This is guaranteed to be a power of two.
  const int64_t num_coeffs = a.size();
  int64_t i = 0;
  for (; i + N <= num_coeffs; i += N) {
    BigInteger* output0_ptr = &output[i];
    BigInteger* output1_ptr = &output[i] + N / 2;
    const Integer* a_ptr = reinterpret_cast<const Integer*>(&a[i]);
    const Integer* b_ptr = reinterpret_cast<const Integer*>(&b[i]);
    auto output0 = hn::Load(d2, output0_ptr);
    auto output1 = hn::Load(d2, output1_ptr);
    auto a_vec = hn::LoadU(d, a_ptr);  // a[i..i+N]
    auto b_vec = hn::LoadU(d, b_ptr);  // b[i..i+N]

    const auto sum = hn::Add(a_vec, b_vec);
    const auto sum_lo = hn::PromoteLowerTo(d2, sum);
    hn::Store(hn::Add(sum_lo, output0), d2, output0_ptr);
    const auto sum_hi = hn::PromoteUpperTo(d2, sum);
    hn::Store(hn::Add(sum_hi, output1), d2, output1_ptr);
  }

  // For the remaining elements in `a` and `b`.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInteger>(a[i].GetMontgomeryRepresentation()) +
                 b[i].GetMontgomeryRepresentation();
  }
}

using ModularInt64 = MontgomeryInt<Uint64>;
using BigInt64 = ModularInt64::BigInt;

// Specialized version for 64-bit integers, as multiplication needs more careful
// treatment.
template <>
void BatchAddMontgomeryRepHwy(absl::Span<const ModularInt64> a,
                              absl::Span<const ModularInt64> b,
                              hwy::AlignedVector<BigInt64>& output) {
  using D = hn::ScalableTag<Uint64>;
  const D d;
  const int N = hn::Lanes(d);
  const int64_t num_coeffs = a.size();

  int64_t i = 0;

  for (; i + N <= num_coeffs; i += N) {
    // Load the lower N/2 and upper N/2 values from `output`.
    Uint64* output0_ptr = reinterpret_cast<Uint64*>(&output[i]);
    Uint64* output1_ptr = reinterpret_cast<Uint64*>(&output[i]) + N;
    auto output0 = hn::Load(d, output0_ptr);
    auto output1 = hn::Load(d, output1_ptr);

    const Uint64* a_ptr = reinterpret_cast<const Uint64*>(&a[i]);
    const Uint64* b_ptr = reinterpret_cast<const Uint64*>(&b[i]);
    auto a_vec = hn::LoadU(d, a_ptr);
    auto b_vec = hn::LoadU(d, b_ptr);

    auto sum_lo = hn::Add(a_vec, b_vec);
    auto carry = hn::Lt(sum_lo, a_vec);
    auto sum_hi = hn::IfThenElseZero(carry, hn::Set(d, 1));
    auto sum0 = hn::InterleaveWholeLower(d, sum_lo, sum_hi);
    auto sum1 = hn::InterleaveWholeUpper(d, sum_lo, sum_hi);

    auto output0_new = hn::Add(output0, sum0);
    auto output1_new = hn::Add(output1, sum1);
    hn::Store(output0_new, d, output0_ptr);
    hn::Store(output1_new, d, output1_ptr);
  }

  // Handle the remaining elements in the input vectors.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInt64>(a[i].GetMontgomeryRepresentation()) +
                 b[i].GetMontgomeryRepresentation();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Sub
////////////////////////////////////////////////////////////////////////////////

template <typename Integer>
void BatchSubMontgomeryRepHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b, Integer q,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  using D = hn::ScalableTag<Integer>;
  using Wide = hwy::MakeWide<hn::TFromD<D>>;
  const D d;                          // For lanes containing Integers.
  const hn::Repartition<Wide, D> d2;  // For double width products of Integers.

  const int N = hn::Lanes(d);  // This is guaranteed to be a power of two.
  const int64_t num_coeffs = a.size();
  auto q_vec = hn::Set(d, q);
  int64_t i = 0;
  for (; i + N <= num_coeffs; i += N) {
    BigInteger* output0_ptr = &output[i];
    BigInteger* output1_ptr = &output[i] + N / 2;
    const Integer* a_ptr = reinterpret_cast<const Integer*>(&a[i]);
    const Integer* b_ptr = reinterpret_cast<const Integer*>(&b[i]);
    auto output0 = hn::Load(d2, output0_ptr);
    auto output1 = hn::Load(d2, output1_ptr);
    auto a_vec = hn::LoadU(d, a_ptr);  // a[i..i+N]
    auto b_vec = hn::LoadU(d, b_ptr);  // b[i..i+N]

    auto underflow_mask = hn::Lt(a_vec, b_vec);
    auto aa_vec = hn::IfThenElse(underflow_mask, hn::Add(a_vec, q_vec), a_vec);

    const auto diff = hn::Sub(aa_vec, b_vec);
    const auto diff_lo = hn::PromoteLowerTo(d2, diff);
    hn::Store(hn::Add(diff_lo, output0), d2, output0_ptr);
    const auto diff_hi = hn::PromoteUpperTo(d2, diff);
    hn::Store(hn::Add(diff_hi, output1), d2, output1_ptr);
  }

  // For the remaining elements in `a` and `b`.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInteger>(a[i].GetMontgomeryRepresentation()) +
                 q - b[i].GetMontgomeryRepresentation();
  }
}

using ModularInt64 = MontgomeryInt<Uint64>;
using BigInt64 = ModularInt64::BigInt;

// Specialized version for 64-bit integers, as multiplication needs more careful
// treatment.
template <>
void BatchSubMontgomeryRepHwy(absl::Span<const ModularInt64> a,
                              absl::Span<const ModularInt64> b, Uint64 q,
                              hwy::AlignedVector<BigInt64>& output) {
  using D = hn::ScalableTag<Uint64>;
  const D d;
  const int N = hn::Lanes(d);
  const int64_t num_coeffs = a.size();
  auto q_vec = hn::Set(d, q);
  auto zero = hn::Zero(d);

  int64_t i = 0;

  for (; i + N <= num_coeffs; i += N) {
    // Load the lower N/2 and upper N/2 values from `output`.
    Uint64* output0_ptr = reinterpret_cast<Uint64*>(&output[i]);
    Uint64* output1_ptr = reinterpret_cast<Uint64*>(&output[i]) + N;
    auto output0 = hn::Load(d, output0_ptr);
    auto output1 = hn::Load(d, output1_ptr);

    const Uint64* a_ptr = reinterpret_cast<const Uint64*>(&a[i]);
    const Uint64* b_ptr = reinterpret_cast<const Uint64*>(&b[i]);
    auto a_vec = hn::LoadU(d, a_ptr);
    auto b_vec = hn::LoadU(d, b_ptr);

    auto underflow_mask = hn::Lt(a_vec, b_vec);
    auto aa_vec = hn::IfThenElse(underflow_mask, hn::Add(a_vec, q_vec), a_vec);

    auto diff = hn::Sub(aa_vec, b_vec);

    auto diff0 = hn::InterleaveWholeLower(d, diff, zero);
    auto diff1 = hn::InterleaveWholeUpper(d, diff, zero);

    auto output0_new = hn::Add(output0, diff0);
    auto output1_new = hn::Add(output1, diff1);
    hn::Store(output0_new, d, output0_ptr);
    hn::Store(output1_new, d, output1_ptr);
  }

  // Handle the remaining elements in the input vectors.
  for (; i < num_coeffs; ++i) {
    output[i] += static_cast<BigInt64>(a[i].GetMontgomeryRepresentation()) + q -
                 b[i].GetMontgomeryRepresentation();
  }
}

#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace HWY_NAMESPACE
}  // namespace rlwe::internal
HWY_AFTER_NAMESPACE();

#if HWY_ONCE || HWY_IDE
namespace rlwe::internal {

template <typename Integer>
void BatchFusedMulAddMontgomeryRepNoHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  for (int j = 0; j < a.size(); ++j) {
    output[j] += static_cast<BigInteger>(a[j].GetMontgomeryRepresentation()) *
                 b[j].GetMontgomeryRepresentation();
  }
}

// For now we only instantiate the Uint32 and the specialized Uint64 versions,
// as they are the most common integer types used in RNS RLWE schemes.
HWY_EXPORT_T(BatchFusedMulAddMontgomeryRepHwy32,
             BatchFusedMulAddMontgomeryRepHwy<Uint32>);
HWY_EXPORT_T(BatchFusedMulAddMontgomeryRepHwy64,
             BatchFusedMulAddMontgomeryRepHwy<Uint64>);

template <typename T>
void BatchFusedMulAddMontgomeryRep(
    absl::Span<const MontgomeryInt<T>> a, absl::Span<const MontgomeryInt<T>> b,
    hwy::AlignedVector<typename BigInt<T>::value_type>& output) {
  BatchFusedMulAddMontgomeryRepNoHwy(a, b, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchFusedMulAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint32>> a,
    absl::Span<const MontgomeryInt<Uint32>> b,
    hwy::AlignedVector<BigInt<Uint32>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchFusedMulAddMontgomeryRepHwy32)(a, b, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchFusedMulAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint64>> a,
    absl::Span<const MontgomeryInt<Uint64>> b,
    hwy::AlignedVector<BigInt<Uint64>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchFusedMulAddMontgomeryRepHwy64)(a, b, output);
}

////////////////////////////////////////////////////////////////////////////////
// FMSumAdd
////////////////////////////////////////////////////////////////////////////////

template <typename Integer>
void BatchFusedMulSumAddMontgomeryRepNoHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    absl::Span<const MontgomeryInt<Integer>> c,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  for (int j = 0; j < a.size(); ++j) {
    output[j] += static_cast<BigInteger>(a[j].GetMontgomeryRepresentation() +
                                         b[j].GetMontgomeryRepresentation()) *
                 c[j].GetMontgomeryRepresentation();
  }
}

// For now we only instantiate the Uint32 and the specialized Uint64 versions,
// as they are the most common integer types used in RNS RLWE schemes.
HWY_EXPORT_T(BatchFusedMulSumAddMontgomeryRepHwy32,
             BatchFusedMulSumAddMontgomeryRepHwy<Uint32>);
HWY_EXPORT_T(BatchFusedMulSumAddMontgomeryRepHwy64,
             BatchFusedMulSumAddMontgomeryRepHwy<Uint64>);

template <typename T>
void BatchFusedMulSumAddMontgomeryRep(
    absl::Span<const MontgomeryInt<T>> a, absl::Span<const MontgomeryInt<T>> b,
    absl::Span<const MontgomeryInt<T>> c,
    hwy::AlignedVector<typename BigInt<T>::value_type>& output) {
  BatchFusedMulSumAddMontgomeryRepNoHwy(a, b, c, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchFusedMulSumAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint32>> a,
    absl::Span<const MontgomeryInt<Uint32>> b,
    absl::Span<const MontgomeryInt<Uint32>> c,
    hwy::AlignedVector<BigInt<Uint32>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchFusedMulSumAddMontgomeryRepHwy32)(a, b, c,
                                                                output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchFusedMulSumAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint64>> a,
    absl::Span<const MontgomeryInt<Uint64>> b,
    absl::Span<const MontgomeryInt<Uint64>> c,
    hwy::AlignedVector<BigInt<Uint64>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchFusedMulSumAddMontgomeryRepHwy64)(a, b, c,
                                                                output);
}

////////////////////////////////////////////////////////////////////////////////
// Add
////////////////////////////////////////////////////////////////////////////////
template <typename Integer>
void BatchAddMontgomeryRepNoHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  for (int j = 0; j < a.size(); ++j) {
    output[j] += static_cast<BigInteger>(a[j].GetMontgomeryRepresentation()) *
                 b[j].GetMontgomeryRepresentation();
  }
}

// For now we only instantiate the Uint32 and the specialized Uint64 versions,
// as they are the most common integer types used in RNS RLWE schemes.
HWY_EXPORT_T(BatchAddMontgomeryRepHwy32, BatchAddMontgomeryRepHwy<Uint32>);
HWY_EXPORT_T(BatchAddMontgomeryRepHwy64, BatchAddMontgomeryRepHwy<Uint64>);

template <typename T>
void BatchAddMontgomeryRep(
    absl::Span<const MontgomeryInt<T>> a, absl::Span<const MontgomeryInt<T>> b,
    hwy::AlignedVector<typename BigInt<T>::value_type>& output) {
  BatchAddMontgomeryRepNoHwy(a, b, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint32>> a,
    absl::Span<const MontgomeryInt<Uint32>> b,
    hwy::AlignedVector<BigInt<Uint32>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchAddMontgomeryRepHwy32)(a, b, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchAddMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint64>> a,
    absl::Span<const MontgomeryInt<Uint64>> b,
    hwy::AlignedVector<BigInt<Uint64>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchAddMontgomeryRepHwy64)(a, b, output);
}

////////////////////////////////////////////////////////////////////////////////
// Sub
////////////////////////////////////////////////////////////////////////////////
template <typename Integer>
void BatchSubMontgomeryRepNoHwy(
    absl::Span<const MontgomeryInt<Integer>> a,
    absl::Span<const MontgomeryInt<Integer>> b, Integer q,
    hwy::AlignedVector<typename BigInt<Integer>::value_type>& output) {
  using BigInteger = typename BigInt<Integer>::value_type;
  for (int j = 0; j < a.size(); ++j) {
    output[j] += static_cast<BigInteger>(a[j].GetMontgomeryRepresentation()) *
                 b[j].GetMontgomeryRepresentation();
  }
}

// For now we only instantiate the Uint32 and the specialized Uint64 versions,
// as they are the most common integer types used in RNS RLWE schemes.
HWY_EXPORT_T(BatchSubMontgomeryRepHwy32, BatchSubMontgomeryRepHwy<Uint32>);
HWY_EXPORT_T(BatchSubMontgomeryRepHwy64, BatchSubMontgomeryRepHwy<Uint64>);

template <typename T>
void BatchSubMontgomeryRep(
    absl::Span<const MontgomeryInt<T>> a, absl::Span<const MontgomeryInt<T>> b,
    T q, hwy::AlignedVector<typename BigInt<T>::value_type>& output) {
  BatchSubMontgomeryRepNoHwy(a, b, q, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchSubMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint32>> a,
    absl::Span<const MontgomeryInt<Uint32>> b, Uint32 q,
    hwy::AlignedVector<BigInt<Uint32>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchSubMontgomeryRepHwy32)(a, b, q, output);
}

// Specialized instantiation to use the highway version of FMA computation.
template <>
void BatchSubMontgomeryRep(
    absl::Span<const MontgomeryInt<Uint64>> a,
    absl::Span<const MontgomeryInt<Uint64>> b, Uint64 q,
    hwy::AlignedVector<BigInt<Uint64>::value_type>& output) {
  HWY_DYNAMIC_DISPATCH_T(BatchSubMontgomeryRepHwy64)(a, b, q, output);
}

}  // namespace rlwe::internal
#endif  // HWY_ONCE || HWY_IDE
