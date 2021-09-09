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

#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_cpu;
using dal::backend::context_gpu;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::compute>;


template <typename Float>
inline auto compute_means(sycl::queue& queue,
                          const pr::ndview<Float, 2>& data,
                          const pr::ndview<Float, 1>& means,
                          const dal::backend::event_vector& deps = {}) {
    
    const std::int64_t column_count = data.get_dimension(1);
    const std::int64_t row_count = data.get_dimension(0);

    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
    
    const Float inv_n = Float(1.0 / double(row_count));

    const Float* sums_ptr = sums.get_data();
    Float* means_ptr = means.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_multiple_nd_range_1d(p, device_max_wg_size(q));

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const std::int64_t i = id.get_global_id();
            if (i < p) {
                const Float s = sums_ptr[i];
                means_ptr[i] = inv_n * s;
            }
        });
    });
}

template <typename Float>
inline auto compute_covariance(sycl::queue& queue,
                               const pr::ndview<Float, 2>& data,
                               const pr::ndview<Float, 1>& means,
                               const pr::ndview<Float, 2>& covariance,
                               const dal::backend::event_vector& deps = {}) {
    const std::int64_t column_count = data.get_dimension(1);
    const std::int64_t row_count = data.get_dimension(0);

    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
    
    const Float inv_n = Float(1.0 / double(row_count));
    const Float inv_n1 = (row_count > 1.0f) ? Float(1.0 / double(row_count - 1)) : 1.0f;

    const Float* sums_ptr = sums.get_data();
    const Float* corr_ptr = corr.get_mutable_data();
    Float* means_ptr = means.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_multiple_nd_range_1d(p, device_max_wg_size(q));

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const std::int64_t i = id.get_global_id();
            if (i < p) {
                const Float s = sums_ptr[i];
                means_ptr[i] = inv_n * s;
            }
        });
    });
}
template <typename Float, typename Task>
static compute_result<Task> call_daal_kernel(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const table& data) {
    bool is_mean_computed = false;

    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t component_count = data.get_column_count();

    daal_covariance::Parameter daal_parameter;
    daal_parameter.outputMatrixType = daal_covariance::covarianceMatrix;

    dal::detail::check_mul_overflow(component_count, component_count);

    const auto daal_data = interop::convert_to_daal_table(queue, data);

    auto arr_means = array<Float>::empty(queue, component_count, sycl::usm::alloc::device);
    const auto daal_means = interop::convert_to_daal_table(queue, arr_means, 1, component_count);

    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto arr_cov_matrix =
            array<Float>::empty(queue, component_count * component_count, sycl::usm::alloc::device);
        const auto daal_cov_matrix =
            interop::convert_to_daal_table(queue, arr_cov_matrix, component_count, component_count);

        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_cov_matrix.get(),
                                                      daal_means.get(),
                                                      &daal_parameter));
        is_mean_computed = true;

        result.set_cov_matrix(
            homogen_table::wrap(arr_cov_matrix, component_count, component_count));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto arr_cor_matrix =
            array<Float>::empty(queue, component_count * component_count, sycl::usm::alloc::device);
        const auto daal_cor_matrix =
            interop::convert_to_daal_table(queue, arr_cor_matrix, component_count, component_count);
        daal_parameter.outputMatrixType = daal_covariance::correlationMatrix;

        interop::status_to_exception(
            daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                      daal_cor_matrix.get(),
                                                      daal_means.get(),
                                                      &daal_parameter));
        is_mean_computed = true;

        result.set_cor_matrix(
            homogen_table::wrap(arr_cor_matrix, component_count, component_count));
    }
    if (desc.get_result_options().test(result_options::means)) {
        if (!is_mean_computed) {
            auto arr_cov_matrix = array<Float>::empty(queue,
                                                      component_count * component_count,
                                                      sycl::usm::alloc::device);

            const auto daal_cov_matrix = interop::convert_to_daal_table(queue,
                                                                        arr_cov_matrix,
                                                                        component_count,
                                                                        component_count);

            interop::status_to_exception(
                daal_covariance_kernel_t<Float>().compute(daal_data.get(),
                                                          daal_cov_matrix.get(),
                                                          daal_means.get(),
                                                          &daal_parameter));
        }
        result.set_means(homogen_table::wrap(arr_means, 1, component_count));
    }
    return result;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    const auto x = input.get_x();
    const auto y = input.get_y();

    auto& queue = ctx.get_queue();

    const std::int64_t x_row_count = x.get_row_count();
    const std::int64_t y_row_count = y.get_row_count();

    ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());
    dal::detail::check_mul_overflow(x_row_count, y_row_count);

    const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
    const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { x_row_count, y_row_count }, sycl::usm::alloc::device);

    auto compute_rbf_event = compute_rbf(queue, x_nd, y_nd, res_nd, desc.get_sigma());

    const auto res_array = res_nd.flatten(queue, { compute_rbf_event });
    auto res_table = homogen_table::wrap(res_array, x_row_count, y_row_count);

    return result_t{}.set_values(res_table);
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend
