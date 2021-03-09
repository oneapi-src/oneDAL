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

#include <src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans::backend {

using std::int64_t;
using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace daal_kmeans = daal::algorithms::kmeans;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_kmeans_lloyd_dense_ucapi_kernel_t =
    daal_kmeans::internal::KMeansDenseLloydBatchKernelUCAPI<Float>;

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::clustering> {
    infer_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const infer_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        interop::execution_context_guard guard(queue);

        const auto data = input.get_data();

        const int64_t row_count = data.get_row_count();
        const int64_t column_count = data.get_column_count();

        const int64_t cluster_count = params.get_cluster_count();
        const int64_t max_iteration_count = 0;

        daal_kmeans::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count),
                                   dal::detail::integral_cast<std::size_t>(max_iteration_count));
        par.resultsToEvaluate = static_cast<DAAL_UINT64>(daal_kmeans::computeAssignments);

        auto arr_data = row_accessor<const Float>{ data }.pull(queue);
        const auto daal_data =
            interop::convert_to_daal_table(queue, arr_data, row_count, column_count);

        auto arr_initial_centroids =
            row_accessor<const Float>{ input.get_model().get_centroids() }.pull(queue);

        dal::detail::check_mul_overflow(cluster_count, column_count);
        array<Float> arr_centroids =
            array<Float>::empty(queue, cluster_count * column_count, sycl::usm::alloc::device);
        array<Float> arr_objective_function_value =
            array<Float>::empty(queue, 1, sycl::usm::alloc::device);
        array<int> arr_labels = array<int>::empty(queue, row_count, sycl::usm::alloc::device);
        array<int> arr_iteration_count = array<int>::empty(queue, 1, sycl::usm::alloc::device);

        const auto daal_initial_centroids = interop::convert_to_daal_table(queue,
                                                                           arr_initial_centroids,
                                                                           cluster_count,
                                                                           column_count);
        const auto daal_centroids =
            interop::convert_to_daal_table(queue, arr_centroids, cluster_count, column_count);
        const auto daal_labels = interop::convert_to_daal_table(queue, arr_labels, row_count, 1);
        const auto daal_objective_function_value =
            interop::convert_to_daal_table(queue, arr_objective_function_value, 1, 1);
        const auto daal_iteration_count =
            interop::convert_to_daal_table(queue, arr_iteration_count, 1, 1);

        daal::data_management::NumericTable* daal_input[2] = { daal_data.get(),
                                                               daal_initial_centroids.get() };

        daal::data_management::NumericTable* daal_output[4] = { daal_centroids.get(),
                                                                daal_labels.get(),
                                                                daal_objective_function_value.get(),
                                                                daal_iteration_count.get() };

        interop::status_to_exception(
            daal_kmeans_lloyd_dense_ucapi_kernel_t<Float>().compute(daal_input, daal_output, &par));

        array<Float> arr_objective_function_value_host =
            array<Float>::empty(queue, 1, sycl::usm::alloc::host);
        queue.memcpy(arr_objective_function_value.get_mutable_data(),
                     arr_iteration_count.get_data(),
                     sizeof(Float) * arr_objective_function_value.get_count());
        queue.wait_and_throw();

        return infer_result<task::clustering>()
            .set_labels(
                dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build())
            .set_objective_function_value(
                static_cast<double>(arr_objective_function_value_host[0]));
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::clustering>;
template struct infer_kernel_gpu<double, method::by_default, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
