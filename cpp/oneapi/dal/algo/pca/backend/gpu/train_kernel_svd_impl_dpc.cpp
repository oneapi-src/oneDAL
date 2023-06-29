/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using model_t = model<task::dim_reduction>;
using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto compute_eigenvectors_on_host(sycl::queue& q,
                                  pr::ndarray<Float, 2>&& data,
                                  std::int64_t component_count,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvectors_on_host);
    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(data.get_dimension(0) == data.get_dimension(1),
                  "dataelation matrix must be square");
    ONEDAL_ASSERT(data.get_dimension(0) > 0);
    const std::int64_t column_count = data.get_dimension(0);

    auto eigvecs = pr::ndarray<Float, 2>::empty({ component_count, column_count });
    auto eigvals = pr::ndarray<Float, 1>::empty(component_count);

    auto host_data = data.to_host(q, deps);
    pr::sym_eigvals_descending(host_data, component_count, eigvecs, eigvals);

    return std::make_tuple(eigvecs, eigvals);
}

// template <typename Float>
// auto centered_data(sycl::queue& q,
//                    const pr::ndview<Float, 2>& data,
//                    const bk::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(centered_data, q);
//     ONEDAL_ASSERT(data.has_data());
//     ONEDAL_ASSERT(data.get_dimension(1) > 0);
//     const std::int64_t column_count = data.get_dimension(1);
//     const std::int64_t row_count = data.get_row_count();

//     pr::ndarray<Float, 2> data_copy =
//         pr::ndarray<Float, 2>::copy(data, { row_count, column_count });

//     auto event = q.submit([&](sycl::handler& cgh) {
//         auto data_acc = data_copy.get_mutable_data();

//         cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::id<1> idx) {
//             Float sum = 0.0;
//             for (std::int64_t i = 0; i < row_count; i++) {
//                 sum += data_acc[i * column_count + idx[0]];
//             }
//             Float mean = sum / row_count;
//             for (std::int64_t i = 0; i < row_count; i++) {
//                 data_acc[i * column_count + idx[0]] -= mean;
//             }
//         });
//     });

//     return std::make_tuple(data_acc, event);
// }

// template <typename Float>
// auto svd_decomposition(sycl::queue& q,
//                        pr::ndview<Float, 2>& data,
//                        std::size_t row_count,
//                        std::size_t column_count) {
//     auto U = pr::ndarray<Float, 2>::empty({ row_count, row_count });
//     auto S = pr::ndarray<Float, 1>::empty({ column_count });
//     auto V_T = pr::ndarray<Float, 2>::empty({ column_count, column_count });

//     //auto result = mkl::lapack::gesvd(queue, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, deps);
//     Float* data_ptr = data.get_mutable_data();
//     Float* U_ptr = U.get_mutable_data();
//     Float* S_ptr = S.get_mutable_data();
//     Float* V_T_ptr = V_T.get_mutable_data();

//     const std::size_t min_dim = std::min(row_count, column_count);
//     const std::size_t max_dim = std::max(row_count, column_count);
//     const std::size_t num_iters = 100;

//     q.submit([&](sycl::handler& cgh) {
//         // auto data_acc = sycl::accessor(data_ptr, cgh, sycl::read_write);
//         // auto U_acc = sycl::accessor(U_ptr, cgh, sycl::write);
//         // auto S_acc = sycl::accessor(S_ptr, cgh, sycl::write);
//         // auto V_T_acc = sycl::accessor(V_T_ptr, cgh, sycl::write);

//         cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
//             for (std::size_t iter = 0; iter < num_iters; ++iter) {
//                 matrix_multiply(data_acc.get_pointer(),
//                                 data_acc.get_pointer(),
//                                 U_acc.get_pointer(),
//                                 row_count,
//                                 column_count,
//                                 row_count,
//                                 column_count,
//                                 row_count,
//                                 row_count);

//                 singular_value_decomposition(U_acc.get_pointer(),
//                                              min_dim,
//                                              max_dim,
//                                              U_acc.get_pointer(),
//                                              S_acc.get_pointer(),
//                                              V_T_acc.get_pointer());

//                 matrix_multiply(U_acc.get_pointer(),
//                                 S_acc.get_pointer(),
//                                 V_T_acc.get_pointer(),
//                                 row_count,
//                                 min_dim,
//                                 min_dim,
//                                 max_dim,
//                                 row_count,
//                                 column_count);
//             }
//         });
//     });

//     q.wait();

//     return std::make_tuple(U, S, V_T);
// }

template <typename Float>
result_t train_kernel_svd_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();

    ONEDAL_ASSERT(data.get_column_count() > 0);
    std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);
    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0));
        gemm_event.wait_and_throw();
    }

    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [eigvecs, eigvals] =
            compute_eigenvectors_on_host(q_, std::move(xtx), component_count, { gemm_event });
        if (desc.get_result_options().test(result_options::eigenvalues)) {
            result.set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count));
        }

        if (desc.get_deterministic()) {
            sign_flip(eigvecs);
        }
        if (desc.get_result_options().test(result_options::eigenvectors)) {
            const auto model = model_t{}.set_eigenvectors(
                homogen_table::wrap(eigvecs.flatten(), component_count, column_count));
            result.set_model(model);
        }
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL