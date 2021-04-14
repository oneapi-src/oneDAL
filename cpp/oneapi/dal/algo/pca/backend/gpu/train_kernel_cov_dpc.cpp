/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"

namespace oneapi::dal::pca::backend {

namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_cpu;
using dal::backend::context_gpu;
using model_t = model<task::dim_reduction>;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();
    const auto alloc = sycl::usm::alloc::device;

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = get_component_count(desc, data);
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    row_accessor<const Float> data_acc{ data };
    const auto data_arr = data_acc.pull(q, { 0, -1 }, alloc);

    const auto data_nd =
        pr::ndview<Float, 2>::wrap(data_arr.get_data(), { row_count, column_count });
    auto row_sums = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc);

    auto reduce_event =
        pr::reduce_by_columns(q, data_nd, row_sums, pr::sum<Float>{}, pr::identity<Float>{});

    auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc);
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc);
    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc);

    auto corr_event = pr::correlation(q, data, row_sums, corr, means, vars, tmp, { reduce_event });

    auto eigenvals = pr::ndarray<Float, 1>::empty(column_count);
    auto host_corr = corr.to_host(q, { corr_event });
    pr::sym_eigvals_descending(host_corr, eigenvals);
    auto eigenvecs = std::move(host_corr);

    ONEDAL_ASSERT(component_count <= column_count);
    const auto eigenvals_arr = array<Float>::empty(component_count);
    const auto eigenvecs_arr = array<Float>::empty(component_count * column_count);

    dal::backend::copy(eigenvals_arr.get_mutable_data(),
                       eigenvals.get_data(),
                       eigenvals_arr.get_count());
    dal::backend::copy(eigenvecs_arr.get_mutable_data(),
                       eigenvecs.get_data(),
                       eigenvecs_arr.get_count());

    const auto model = model_t{}.set_eigenvectors(
        homogen_table::wrap(eigenvecs_arr, component_count, column_count));

    return result_t{}
        .set_model(model)
        .set_eigenvalues(homogen_table::wrap(eigenvals_arr, 1, component_count))
        .set_means(homogen_table::wrap(means.flatten(q), 1, column_count))
        .set_variances(homogen_table::wrap(vars.flatten(q), 1, column_count));
}

template <typename Float>
struct train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
