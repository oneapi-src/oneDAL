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

#include "oneapi/dal/algo/objective_function/backend/gpu/compute_kernel_dense_batch_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/algo/objective_function/backend/objective_impl.hpp"
// #include "oneapi/dal/backend/primitives/objective_function.hpp"
// #include "oneapi/dal/backend/primitives/stat.hpp"
// #include "oneapi/dal/backend/primitives/blas.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::objective_function::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
//using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
result_t compute_kernel_dense_batch_impl<Float>::operator()(const detail::descriptor_base<task_t>& desc,
                                                      const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    ONEDAL_ASSERT(input.get_parameters().has_data());
    ONEDAL_ASSERT(input.get_responses().has_data());
    const auto data = input.get_data();
    const auto params = input.get_parameters();
    const auto responses = input.get_responses();
    const std::int64_t n = data.get_row_count();
    const std::int64_t p = data.get_column_count();
    ONEDAL_ASSERT(responses.get_row_count() == 1);
    ONEDAL_ASSERT(responses.get_column_count() == n);
    ONEDAL_ASSERT(params.get_row_count() == 1);
    ONEDAL_ASSERT(params.get_column_count() == p + 1);

    auto obj_impl = detail::get_objective_impl(desc);

    const double L1 = obj_impl->get_l1_regularization_coefficient();
    const double L2 = obj_impl->get_l2_regularization_coefficient();

    std::cout << L1 << " " << L2 << std::endl;

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device); // might throw when table data is big
    const auto params_nd = pr::table2ndarray_1d<Float>(q_, params, alloc::device);
    const auto responses_nd = pr::table2ndarray_1d<Float>(q_, responses, alloc::device);
    
    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    auto probabilities = pr::ndarray<float_t, 1>::empty(q_, {n + 1}, sycl::usm::alloc::device);
    
    //sycl::event prob_e = compute_probabilities(q_, params_nd, data_nd, probabilities, {});

    if (desc.get_result_options().test(result_options::hessian)) {
        //prob_e.wait_and_throw();
        /*
        auto [out_hessian, out_hess_e] = ndarray<float_t, 2>::zeros(this->get_queue(), { p + 1, p + 1 }, sycl::usm::alloc::device);
        auto hes_event = pr::compute_hessian(this->get_queue(),
                                            params_nd,
                                            data_nd,
                                            responses_nd,
                                            probabilities,
                                            out_hessian,
                                            L1,
                                            L2,
                                            { prob_e, out_hess_e });
        result.set_hessian(homogen_table::wrap(out_hessian.flatten(q_, { hes_event }), p + 1, p + 1));
        */
    }
    return result;
    
}

template class compute_kernel_dense_batch_impl<float>;
template class compute_kernel_dense_batch_impl<double>;

} // namespace oneapi::dal::objective_function::backend

#endif // ONEDAL_DATA_PARALLEL
