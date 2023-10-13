// /*******************************************************************************
// * Copyright 2021 Intel Corporation
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *******************************************************************************/

// #include "oneapi/dal/backend/primitives/lapack/eigen.hpp"
// #include "oneapi/dal/backend/primitives/lapack/syevd.hpp"

// namespace oneapi::dal::backend::primitives {

// template <typename Float>
// sycl::event sym_eigvals_impl(sycl::queue& q,
//                              Float* a,
//                              std::int64_t n,
//                              std::int64_t lda,
//                              Float* w,
//                              const event_vector& deps) {
//     ONEDAL_ASSERT(a);
//     ONEDAL_ASSERT(w);
//     ONEDAL_ASSERT(n > 0);
//     ONEDAL_ASSERT(lda >= n);

//     const std::int64_t lwork = 2 * n * n + 6 * n + 1;
//     const std::int64_t liwork = 5 * n + 3;

//     ONEDAL_ASSERT(lwork > n);
//     ONEDAL_ASSERT(liwork > n);

//     const auto work = ndarray<Float, 1>::empty(lwork);

//     Float* work_ptr = work.get_mutable_data();

//     auto event = mkl::lapack::syevd(q, mkl::job::vec, mkl::uplo::upper, n, a, lda, w, work_ptr, lwork, deps);

//     return event;
// }

// template <typename Float>
// sycl::event flip_eigvals_impl(sycl::queue& q,
//                               Float* a,
//                               Float* w,
//                               std::int64_t n,
//                               std::int64_t lda,
//                               std::int64_t w_count,
//                               Float* a_flipped,
//                               std::int64_t lda_flipped,
//                               Float* w_flipped,
//                               const event_vector& deps) {
//     ONEDAL_ASSERT(a);
//     ONEDAL_ASSERT(w);
//     ONEDAL_ASSERT(a_flipped);
//     ONEDAL_ASSERT(w_flipped);
//     ONEDAL_ASSERT(n > 0);
//     ONEDAL_ASSERT(lda >= n);
//     ONEDAL_ASSERT(w_count > 0);
//     ONEDAL_ASSERT(w_count <= n);

//     return q.submit([&](cl::sycl::handler& cgh) {
//         auto a_accessor = a;
//         auto w_accessor = w;
//         auto a_flipped_accessor = a_flipped;
//         auto w_flipped_accessor = w_flipped;

//         cgh.depends_on(deps);

//         cgh.parallel_for(cl::sycl::range<1>(n / 2), [=](cl::sycl::id<1> idx) {
//             const std::int64_t i = idx[0];
//             const std::int64_t src_i = n - i - 1;
//             const std::int64_t dst_i = i;

//             if (a == a_flipped) {
//                 for (std::int64_t j = 0; j < n; j++) {
//                     std::swap(a_accessor[src_i * lda + j], a_accessor[dst_i * lda + j]);
//                 }
//             }
//             else {
//                 for (std::int64_t j = 0; j < n; j++) {
//                     a_flipped_accessor[dst_i * lda_flipped + j] = a_accessor[src_i * lda + j];
//                 }
//             }

//             if (w == w_flipped) {
//                 std::swap(w_accessor[src_i], w_accessor[dst_i]);
//             }
//             else {
//                 w_flipped_accessor[dst_i] = w_accessor[src_i];
//             }
//         });
//     });
// }

// #define INSTANTIATE_SYM(Float)                                 \
//     template sycl::event sym_eigvals_impl<Float>(sycl::queue&, \
//                                                  Float*,       \
//                                                  std::int64_t, \
//                                                  std::int64_t, \
//                                                  Float*,       \
//                                                  const event_vector&);

// INSTANTIATE_SYM(float)
// INSTANTIATE_SYM(double)

// #define INSTANTIATE_FLIP(Float)                                 \
//     template sycl::event flip_eigvals_impl<Float>(sycl::queue&, \
//                                                   Float*,       \
//                                                   Float*,       \
//                                                   std::int64_t, \
//                                                   std::int64_t, \
//                                                   std::int64_t, \
//                                                   Float*,       \
//                                                   std::int64_t, \
//                                                   Float*,       \
//                                                   const event_vector&);

// INSTANTIATE_FLIP(float)
// INSTANTIATE_FLIP(double)

// } // namespace oneapi::dal::backend::primitives
