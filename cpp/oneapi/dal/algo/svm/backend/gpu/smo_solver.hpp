/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"
// #include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

template <typename Float>
class smo_solver {
public:
    smo_solver(const sycl::queue& queue,
               std::int64_t n_vectors,
               std::int64_t n_ws,
               std::int64_t max_inner_iter,
               Float C,
               Float eps,
               Float tau)
            : queue_(queue),
              n_vectors_(n_vectors),
              n_ws_(n_ws),
              max_inner_iter_(max_inner_iter),
              C_(C),
              eps_(eps),
              tau_(tau) {
        ONEDAL_ASSERT(n_vectors_ > 0);
        ONEDAL_ASSERT(n_vectors_ <= dal::detail::limits<std::uint32_t>::max());
        ONEDAL_ASSERT(max_inner_iter_ > 0);
        ONEDAL_ASSERT(max_inner_iter_ <= dal::detail::limits<std::uint32_t>::max());
        ONEDAL_ASSERT(n_ws_ > 0);
        ONEDAL_ASSERT(n_ws_ <= dal::detail::limits<std::uint32_t>::max());
    }

    sycl::event solve(const pr::ndview<Float, 1>& y,
                      const pr::ndview<Float, 1>& kernel_ws_rows,
                      const pr::ndview<std::uint32_t, 1>& ws_indices,
                      const pr::ndview<Float, 1>& f,
                      pr::ndview<Float, 1>& alpha,
                      pr::ndview<Float, 1>& delta_alpha,
                      pr::ndview<Float, 1>& res_info,
                      const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(ws_indices.get_dimension(0) == n_ws_);
        ONEDAL_ASSERT(delta_alpha.get_dimension(0) == n_ws_);
        ONEDAL_ASSERT(res_info.get_dimension(0) == 2);
        ONEDAL_ASSERT(y.get_dimension(0) == f.get_dimension(0));
        ONEDAL_ASSERT(alpha.get_dimension(0) == f.get_dimension(0));

        const Float* y_ptr = y.get_data();
        const Float* kernel_ws_rows_ptr = kernel_ws_rows.get_data();
        const std::uint32_t* ws_indices_ptr = ws_indices.get_data();
        const Float* f_ptr = f.get_data();
        Float* alpha_ptr = alpha.get_mutable_data();
        Float* delta_alpha_ptr = delta_alpha.get_mutable_data();
        Float* res_info_ptr = res_info.get_mutable_data();

        const sycl::nd_range<1> nd_range = make_multiple_nd_range_1d(n_ws_, n_ws_);
    }

private:
    sycl::queue queue_;

    std::int64_t n_vectors_;
    std::int64_t n_ws_;
    std::int64_t max_inner_iter_;

    Float C_;
    Float eps_;
    Float tau_;

}; // namespace oneapi::dal::svm::backend

#define INSTANTIATE_SMO_SOLVER(F) template class ONEDAL_EXPORT smo_solver<F>;

INSTANTIATE_SMO_SOLVER(float);
INSTANTIATE_SMO_SOLVER(double);

} // namespace oneapi::dal::svm::backend