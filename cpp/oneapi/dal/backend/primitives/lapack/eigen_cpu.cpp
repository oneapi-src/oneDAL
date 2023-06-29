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

// #include "oneapi/dal/backend/dispatcher.hpp"
// #include "oneapi/dal/backend/primitives/lapack/eigen.hpp"

// namespace oneapi::dal::backend::primitives {

// sycl::event flip_eigvals_impl_gpu(sycl::queue& q,
//                                       float* a,
//                                       float* w,
//                                       std::int64_t n,
//                                       std::int64_t lda,
//                                       std::int64_t w_count,
//                                       float* a_flipped,
//                                       std::int64_t lda_flipped,
//                                       float* w_flipped,
//                                       const event_vector& dep) {
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
//             } else {
//                 for (std::int64_t j = 0; j < n; j++) {
//                     a_flipped_accessor[dst_i * lda_flipped + j] = a_accessor[src_i * lda + j];
//                 }
//             }

//             if (w == w_flipped) {
//                 std::swap(w_accessor[src_i], w_accessor[dst_i]);
//             } else {
//                 w_flipped_accessor[dst_i] = w_accessor[src_i];
//             }
//         });
//     });
// }

// #define INSTANTIATE(Float)                                   \
//     template sycl::event flip_eigvals_impl_gpu<Float>(
//         sycl::queue&, \
//         Float*,       \
//                                                     Float*,       \
//                                                     std::int64_t, \
//                                                     std::int64_t, \
//                                                     std::int64_t, \
//                                                     Float*,       \
//                                                     std::int64_t, \
//                                                     Float*);

// INSTANTIATE(float)
// INSTANTIATE(double)

// } // namespace oneapi::dal::backend::primitives
