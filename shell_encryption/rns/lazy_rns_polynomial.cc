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

#include "shell_encryption/rns/lazy_rns_polynomial.h"

#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "shell_encryption/integral_types.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/rns/rns_polynomial_hwy.h"
#include "shell_encryption/status_macros.h"

namespace rlwe {

using ModularInt32 = MontgomeryInt<Uint32>;
using ModularInt64 = MontgomeryInt<Uint64>;

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::CheckFusedMulAddInPlaceParameters(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (!a.IsNttForm() || !b.IsNttForm()) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must be in NTT form.");
  }
  int num_moduli = moduli.size();
  if (a.NumModuli() != num_moduli || b.NumModuli() != num_moduli ||
      coeff_vectors_.size() != num_moduli) {
    return absl::InvalidArgumentError(
        "Polynomials `a`, `b`, and this must all be defined wrt `moduli`");
  }
  int num_coeffs = coeff_vectors_[0].size();
  if (a.NumCoeffs() != num_coeffs || b.NumCoeffs() != num_coeffs) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must have the same number of coefficients as "
        "this lazy polynomial.");
  }
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status
LazyRnsPolynomial<ModularInt>::CheckFusedMulSumAddInPlaceParameters(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    const RnsPolynomial<ModularInt>& c,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (!a.IsNttForm() || !b.IsNttForm() || !c.IsNttForm()) {
    return absl::InvalidArgumentError(
        "Polynomials `a`, `b`, and `c` must be in NTT form.");
  }
  int num_moduli = moduli.size();
  if (a.NumModuli() != num_moduli || b.NumModuli() != num_moduli ||
      c.NumModuli() != num_moduli || coeff_vectors_.size() != num_moduli) {
    return absl::InvalidArgumentError(
        "Polynomials `a`, `b`, `c`, and this must all be defined wrt `moduli`");
  }
  int num_coeffs = coeff_vectors_[0].size();
  if (a.NumCoeffs() != num_coeffs || b.NumCoeffs() != num_coeffs ||
      c.NumCoeffs() != num_coeffs) {
    return absl::InvalidArgumentError(
        "Polynomials `a`, `b`, and `c` must have the same number of "
        "coefficients as "
        "this lazy polynomial.");
  }
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::FusedMulAddInPlace(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulAddInPlaceParameters(a, b, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  int num_coeffs = coeff_vectors_[0].size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors_[i][j] +=
          static_cast<BigInt>(
              a_coeff_vectors[i][j].GetMontgomeryRepresentation()) *
          b_coeff_vectors[i][j].GetMontgomeryRepresentation();
    }
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt32>::FusedMulAddInPlace(
    const RnsPolynomial<ModularInt32>& a, const RnsPolynomial<ModularInt32>& b,
    absl::Span<const PrimeModulus<ModularInt32>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulAddInPlaceParameters(a, b, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    internal::BatchFusedMulAddMontgomeryRep<Uint32>(
        a_coeff_vectors[i], b_coeff_vectors[i], coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt64>::FusedMulAddInPlace(
    const RnsPolynomial<ModularInt64>& a, const RnsPolynomial<ModularInt64>& b,
    absl::Span<const PrimeModulus<ModularInt64>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulAddInPlaceParameters(a, b, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }
  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    internal::BatchFusedMulAddMontgomeryRep<Uint64>(
        a_coeff_vectors[i], b_coeff_vectors[i], coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::FusedMulAddInPlace(
    const LazyRnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }
  int num_moduli = moduli.size();
  int num_coeffs = coeff_vectors_[0].size();
  const auto& b_coeff_vectors = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors_[i][j] +=
          a.coeff_vectors_[i][j] *
          b_coeff_vectors[i][j].GetMontgomeryRepresentation();
    }
  }
  current_level_++;
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::FusedMulSumAddInPlace(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    const RnsPolynomial<ModularInt>& c,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  int num_coeffs = coeff_vectors_[0].size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors_[i][j] +=
          static_cast<BigInt>(
              a_coeff_vectors[i][j].GetMontgomeryRepresentation() +
              b_coeff_vectors[i][j].GetMontgomeryRepresentation()) *
          c_coeff_vectors[i][j].GetMontgomeryRepresentation();
    }
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt32>::FusedMulSumAddInPlace(
    const RnsPolynomial<ModularInt32>& a, const RnsPolynomial<ModularInt32>& b,
    const RnsPolynomial<ModularInt32>& c,
    absl::Span<const PrimeModulus<ModularInt32>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    internal::BatchFusedMulSumAddMontgomeryRep<Uint32>(
        a_coeff_vectors[i], b_coeff_vectors[i], c_coeff_vectors[i],
        coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt64>::FusedMulSumAddInPlace(
    const RnsPolynomial<ModularInt64>& a, const RnsPolynomial<ModularInt64>& b,
    const RnsPolynomial<ModularInt64>& c,
    absl::Span<const PrimeModulus<ModularInt64>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    internal::BatchFusedMulSumAddMontgomeryRep<Uint64>(
        a_coeff_vectors[i], b_coeff_vectors[i], c_coeff_vectors[i],
        coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::FusedMulDifferenceAddInPlace(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    const RnsPolynomial<ModularInt>& c,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  int num_coeffs = coeff_vectors_[0].size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    const auto qi = moduli[i]->Modulus();
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors_[i][j] +=
          static_cast<BigInt>(
              a_coeff_vectors[i][j].GetMontgomeryRepresentation() + qi -
              b_coeff_vectors[i][j].GetMontgomeryRepresentation()) *
          c_coeff_vectors[i][j].GetMontgomeryRepresentation();
    }
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt32>::FusedMulDifferenceAddInPlace(
    const RnsPolynomial<ModularInt32>& a, const RnsPolynomial<ModularInt32>& b,
    const RnsPolynomial<ModularInt32>& c,
    absl::Span<const PrimeModulus<ModularInt32>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    const auto qi = moduli[i]->Modulus();
    internal::BatchFusedMulDifferenceAddMontgomeryRep<Uint32>(
        a_coeff_vectors[i], b_coeff_vectors[i], c_coeff_vectors[i], qi,
        coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <>
absl::Status LazyRnsPolynomial<ModularInt64>::FusedMulDifferenceAddInPlace(
    const RnsPolynomial<ModularInt64>& a, const RnsPolynomial<ModularInt64>& b,
    const RnsPolynomial<ModularInt64>& c,
    absl::Span<const PrimeModulus<ModularInt64>* const> moduli) {
  RLWE_RETURN_IF_ERROR(CheckFusedMulSumAddInPlaceParameters(a, b, c, moduli));
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }

  int num_moduli = moduli.size();
  const auto& a_coeff_vectors = a.Coeffs();
  const auto& b_coeff_vectors = b.Coeffs();
  const auto& c_coeff_vectors = c.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    const auto qi = moduli[i]->Modulus();
    internal::BatchFusedMulDifferenceAddMontgomeryRep<Uint64>(
        a_coeff_vectors[i], b_coeff_vectors[i], c_coeff_vectors[i], qi,
        coeff_vectors_[i]);
  }
  current_level_++;
  return absl::OkStatus();
}

template <typename ModularInt>
absl::Status LazyRnsPolynomial<ModularInt>::MulInPlace(
    const RnsPolynomial<ModularInt>& a,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (current_level_ == maximum_level_) {
    Refresh(moduli);
  }
  int num_moduli = moduli.size();
  int num_coeffs = coeff_vectors_[0].size();
  const auto& a_coeff_vectors = a.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors_[i][j] *= static_cast<BigInt>(
          a_coeff_vectors[i][j].GetMontgomeryRepresentation());
    }
  }
  current_level_++;
  return absl::OkStatus();
}

template <typename ModularInt>
absl::StatusOr<LazyRnsPolynomial<ModularInt>>
LazyRnsPolynomial<ModularInt>::CreateFromSum(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (a.IsNttForm() ^ b.IsNttForm()) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must be in the same form.");
  }
  int num_coeffs = a.NumCoeffs();
  if (b.NumCoeffs() != num_coeffs) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must have the same number of coefficients.");
  }
  int num_moduli = moduli.size();
  if (a.NumModuli() != num_moduli || b.NumModuli() != num_moduli) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must be defined wrt `moduli`");
  }

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    const auto& a_coeffs = a.Coeffs()[i];
    const auto& b_coeffs = b.Coeffs()[i];
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors[i][j] =
          static_cast<BigInt>(a_coeffs[j].GetMontgomeryRepresentation() +
                              b_coeffs[j].GetMontgomeryRepresentation());
    }
  }

  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template <>
absl::StatusOr<LazyRnsPolynomial<ModularInt32>>
LazyRnsPolynomial<ModularInt32>::CreateFromSum(
    const RnsPolynomial<ModularInt32>& a, const RnsPolynomial<ModularInt32>& b,
    absl::Span<const PrimeModulus<ModularInt32>* const> moduli) {
  int num_coeffs = a.NumCoeffs();
  int num_moduli = moduli.size();

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  const auto& a_coeffs = a.Coeffs();
  const auto& b_coeffs = b.Coeffs();
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    internal::BatchAddMontgomeryRep<Uint32>(a_coeffs[i], b_coeffs[i],
                                            coeff_vectors[i]);
  }
  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template <>
absl::StatusOr<LazyRnsPolynomial<ModularInt64>>
LazyRnsPolynomial<ModularInt64>::CreateFromSum(
    const RnsPolynomial<ModularInt64>& a, const RnsPolynomial<ModularInt64>& b,
    absl::Span<const PrimeModulus<ModularInt64>* const> moduli) {
  int num_coeffs = a.NumCoeffs();
  int num_moduli = moduli.size();

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  const auto& a_coeffs = a.Coeffs();
  const auto& b_coeffs = b.Coeffs();
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    internal::BatchAddMontgomeryRep<Uint64>(a_coeffs[i], b_coeffs[i],
                                            coeff_vectors[i]);
  }

  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template <typename ModularInt>
absl::StatusOr<LazyRnsPolynomial<ModularInt>>
LazyRnsPolynomial<ModularInt>::CreateFromDifference(
    const RnsPolynomial<ModularInt>& a, const RnsPolynomial<ModularInt>& b,
    absl::Span<const PrimeModulus<ModularInt>* const> moduli) {
  if (a.IsNttForm() ^ b.IsNttForm()) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must be in the same form.");
  }
  int num_coeffs = a.NumCoeffs();
  if (b.NumCoeffs() != num_coeffs) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must have the same number of coefficients.");
  }
  int num_moduli = moduli.size();
  if (a.NumModuli() != num_moduli || b.NumModuli() != num_moduli) {
    return absl::InvalidArgumentError(
        "Polynomials `a` and `b` must be defined wrt `moduli`");
  }

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    const auto& a_coeffs = a.Coeffs()[i];
    const auto& b_coeffs = b.Coeffs()[i];
    const auto qi = moduli[i]->Modulus();
    for (int j = 0; j < num_coeffs; ++j) {
      coeff_vectors[i][j] =
          static_cast<BigInt>(a_coeffs[j].GetMontgomeryRepresentation()) + qi -
          b_coeffs[j].GetMontgomeryRepresentation();
    }
  }

  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template <>
absl::StatusOr<LazyRnsPolynomial<ModularInt32>>
LazyRnsPolynomial<ModularInt32>::CreateFromDifference(
    const RnsPolynomial<ModularInt32>& a, const RnsPolynomial<ModularInt32>& b,
    absl::Span<const PrimeModulus<ModularInt32>* const> moduli) {
  int num_coeffs = a.NumCoeffs();
  int num_moduli = moduli.size();

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  const auto& a_coeffs = a.Coeffs();
  const auto& b_coeffs = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    const auto qi = moduli[i]->Modulus();
    internal::BatchSubMontgomeryRep<Uint32>(a_coeffs[i], b_coeffs[i], qi,
                                            coeff_vectors[i]);
  }

  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template <>
absl::StatusOr<LazyRnsPolynomial<ModularInt64>>
LazyRnsPolynomial<ModularInt64>::CreateFromDifference(
    const RnsPolynomial<ModularInt64>& a, const RnsPolynomial<ModularInt64>& b,
    absl::Span<const PrimeModulus<ModularInt64>* const> moduli) {
  int num_coeffs = a.NumCoeffs();
  int num_moduli = moduli.size();

  // Get the maximal number of FMA operations can be made wrt the RNS moduli.
  RLWE_ASSIGN_OR_RETURN(int log_maximum_level, ComputeLogMaximalLevel(moduli));

  // Let's add the CRT coefficients.
  std::vector<hwy::AlignedVector<BigInt>> coeff_vectors(num_moduli);
  const auto& a_coeffs = a.Coeffs();
  const auto& b_coeffs = b.Coeffs();
  for (int i = 0; i < num_moduli; ++i) {
    coeff_vectors[i].resize(num_coeffs);
    const auto qi = moduli[i]->Modulus();
    internal::BatchSubMontgomeryRep<Uint64>(a_coeffs[i], b_coeffs[i], qi,
                                            coeff_vectors[i]);
  }

  return LazyRnsPolynomial(std::move(coeff_vectors),
                           /*current_level=*/static_cast<BigInt>(1),
                           static_cast<BigInt>(1) << log_maximum_level);
}

template class LazyRnsPolynomial<MontgomeryInt<Uint16>>;
template class LazyRnsPolynomial<MontgomeryInt<Uint32>>;
template class LazyRnsPolynomial<MontgomeryInt<Uint64>>;
template class LazyRnsPolynomial<MontgomeryInt<absl::uint128>>;
#ifdef ABSL_HAVE_INTRINSIC_INT128
template class LazyRnsPolynomial<MontgomeryInt<unsigned __int128>>;
#endif

}  // namespace rlwe
