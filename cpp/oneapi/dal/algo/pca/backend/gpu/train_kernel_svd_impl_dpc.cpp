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

//TODO::call wrapper instead of direct call of mkl
template <typename Float>
auto svd_decomposition(sycl::queue& queue, pr::ndview<Float, 2>& data) {
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);
    std::cout << "here 1" << std::endl;
    auto U = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);
    std::cout << "here 2" << std::endl;
    auto S = pr::ndarray<Float, 1>::empty(queue, { row_count }, alloc::device);
    std::cout << "here 3" << std::endl;
    auto V_T = pr::ndarray<Float, 2>::empty(queue, { row_count, row_count }, alloc::device);
    std::cout << "here 4" << std::endl;

    Float* data_ptr = data.get_mutable_data();
    Float* U_ptr = U.get_mutable_data();
    Float* S_ptr = S.get_mutable_data();
    Float* V_T_ptr = V_T.get_mutable_data();
    std::int64_t lda = data.get_leading_stride();
    std::int64_t ldu = U.get_leading_stride();
    std::int64_t ldvt = V_T.get_leading_stride();

    std::cout << "here 5" << std::endl;
    {
        ONEDAL_PROFILER_TASK(gesvd, queue);
        auto event = pr::gesvd<mkl::jobsvd::vectors, mkl::jobsvd::somevec>(queue,
                                                                           row_count,
                                                                           column_count,
                                                                           data_ptr,
                                                                           lda,
                                                                           S_ptr,
                                                                           U_ptr,
                                                                           ldu,
                                                                           V_T_ptr,
                                                                           ldvt,
                                                                           {});
    }
    std::cout << "here 6" << std::endl;
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
    if (data.get_data_layout() == data_layout::row_major) {
        std::cout << "row major" << std::endl;
    }
    else if (data.get_data_layout() == data_layout::column_major) {
        std::cout << "column major" << std::endl;
    }
    pr::ndview<Float, 2> data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);
    //TODO: add mean centering by default

    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [U, S, V_T] = svd_decomposition(q_, data_nd);
        std::cout << "here 10" << std::endl;
        if (desc.get_result_options().test(result_options::eigenvalues)) {
            result.set_eigenvalues(homogen_table::wrap(S.flatten(q_), 1, row_count));
        }
        //TODO: fix bug with sign flip function(move computations on gpu)
        std::cout << "here 11" << std::endl;
        auto u_host = U.to_host(q_);
        if (desc.get_deterministic()) {
            sign_flip(u_host);
        }
        std::cout << "here 12" << std::endl;
        if (desc.get_result_options().test(result_options::eigenvectors)) {
            const auto model = model_t{}.set_eigenvectors(
                homogen_table::wrap(u_host.flatten(), column_count, column_count));
            result.set_model(model);
        }
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend
