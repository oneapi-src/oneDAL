/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/src/algorithms/pca/transform/oneapi/pca_transform_dense_default_batch_oneapi.h>
#include <daal/include/algorithms/pca/transform/pca_transform_types.h>

#include "oneapi/dal/algo/pca/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_gpu;
using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = descriptor_base<task::dim_reduction>;

namespace daal_pca_tr = daal::algorithms::pca::transform;
namespace daal_pca_tr_oneapi = daal::algorithms::pca::transform::oneapi;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_pca_transform_oneapi_kernel_t =
    daal_pca_tr_oneapi::internal::TransformKernelOneAPI<Float, daal_pca_tr::Method::defaultDense>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const model_t& model) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = desc.get_component_count();

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    auto arr_eigvec = row_accessor<const Float>{ model.get_eigenvectors() }.pull(queue);

    dal::detail::check_mul_overflow(row_count, component_count);
    auto arr_result = array<Float>::empty(queue, row_count * component_count);

    // TODO: read-only access performed with deep copy of data since daal numeric tables are mutable.
    // Need to create special immutable homogen table on daal interop side

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_eigenvectors = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                               arr_eigvec,
                                                                               component_count,
                                                                               column_count);
    const auto daal_result =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_result, row_count, component_count);

    interop::status_to_exception(
        daal_pca_transform_oneapi_kernel_t<Float>()
            .compute(*daal_data, *daal_eigenvectors, nullptr, nullptr, nullptr, *daal_result));

    return result_t{}.set_transformed_data(
        dal::detail::homogen_table_builder{}.reset(arr_result, row_count, component_count).build());
}

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float>
struct infer_kernel_gpu<Float, task::dim_reduction> {
    infer_result<task::dim_reduction> operator()(
        const dal::backend::context_gpu& ctx,
        const descriptor_base<task::dim_reduction>& desc,
        const infer_input<task::dim_reduction>& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::dim_reduction>;
template struct infer_kernel_gpu<double, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
