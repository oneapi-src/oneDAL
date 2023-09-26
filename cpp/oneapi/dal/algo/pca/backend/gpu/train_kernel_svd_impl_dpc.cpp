/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel_svd_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include <iostream>
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace mkl = oneapi::fpk;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using model_t = model<task::dim_reduction>;
using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto svd_decomposition(sycl::queue& queue, pr::ndview<Float, 2>& data) {
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto U = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);
    auto S = pr::ndarray<Float, 1>::empty(queue, { row_count }, alloc::device);
    auto V_T = pr::ndarray<Float, 2>::empty(queue, { row_count, row_count }, alloc::device);
    Float* data_ptr = data.get_mutable_data();
    Float* U_ptr = U.get_mutable_data();
    Float* S_ptr = S.get_mutable_data();
    Float* V_T_ptr = V_T.get_mutable_data();
    std::int64_t lda = column_count;
    std::int64_t ldu = column_count;
    std::int64_t ldvt = row_count;
    const auto scratchpad_size = mkl::lapack::gesvd_scratchpad_size<Float>(queue,
                                                                           mkl::jobsvd::vectors,
                                                                           mkl::jobsvd::vectors,
                                                                           column_count,
                                                                           row_count,
                                                                           lda,
                                                                           ldu,
                                                                           ldvt);
    auto scratchpad =
        pr::ndarray<Float, 1>::empty(queue, { scratchpad_size }, sycl::usm::alloc::device);
    auto scratchpad_ptr = scratchpad.get_mutable_data();
    auto event = mkl::lapack::gesvd(queue,
                                    mkl::jobsvd::vectors,
                                    mkl::jobsvd::vectors,
                                    column_count,
                                    row_count,
                                    data_ptr,
                                    lda,
                                    S_ptr,
                                    U_ptr,
                                    ldu,
                                    V_T_ptr,
                                    ldvt,
                                    scratchpad_ptr,
                                    scratchpad_size,
                                    {});
    return std::make_tuple(U, S, V_T);
}

template <typename Float>
result_t train_kernel_svd_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();

    ONEDAL_ASSERT(data.get_column_count() > 0);
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    pr::ndview<Float, 2> data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);
    //TODO: add mean centering by default
    //sycl::event mean_center_event;
    // {
    //     ONEDAL_PROFILER_TASK(elementwise_difference, q_);
    //     mean_center_event = pr::elementwise_difference(q_, row_count, data_nd, means_nd,
    //                                                    mean_centered_data_nd);
    // }

    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [U, S, V_T] = svd_decomposition(q_, data_nd);

        if (desc.get_result_options().test(result_options::eigenvalues)) {
            result.set_eigenvalues(homogen_table::wrap(S.flatten(q_), 1, row_count));
        }
        //TODO: fix bug with sign flip function(move computations on gpu)
        // if (desc.get_deterministic()) {
        //     sign_flip(U);
        // }

        if (desc.get_result_options().test(result_options::eigenvectors)) {
            const auto model = model_t{}.set_eigenvectors(
                homogen_table::wrap(U.flatten(q_), column_count, column_count));
            result.set_model(model);
        }
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend
