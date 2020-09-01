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

#define DAAL_SYCL_INTERFACE
#define DAAL_SYCL_INTERFACE_USM
#define DAAL_SYCL_INTERFACE_REVERSED_RANGE

#include <src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans::backend {

using std::int64_t;
using dal::backend::context_gpu;

namespace daal_kmeans = daal::algorithms::kmeans;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_kmeans_lloyd_dense_ucapi_kernel_t =
    daal_kmeans::internal::KMeansDenseLloydBatchKernelUCAPI<Float>;

template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_dense> {
    train_result operator()(const dal::backend::context_gpu& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        if (!(input.get_initial_centroids().has_data())) {
            throw domain_error("Input initial_centroids should not be empty");
        }

        auto& queue = ctx.get_queue();
        interop::execution_context_guard guard(queue);

        const auto data = input.get_data();

        const int64_t row_count = data.get_row_count();
        const int64_t column_count = data.get_column_count();

        const int64_t cluster_count = params.get_cluster_count();
        const int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();

        daal_kmeans::Parameter par(cluster_count, max_iteration_count);
        par.accuracyThreshold = accuracy_threshold;

        auto arr_data = row_accessor<const Float>{ data }.pull(queue);
        const auto daal_data = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                           arr_data,
                                                                           data.get_row_count(),
                                                                           data.get_column_count());

        auto arr_initial_centroids =
            row_accessor<const Float>{ input.get_initial_centroids() }.pull(queue);

        array<Float> arr_centroids = array<Float>::empty(queue, cluster_count * column_count);
        array<int> arr_labels = array<int>::empty(queue, row_count);
        array<Float> arr_objective_function_value = array<Float>::empty(queue, 1);
        array<int> arr_iteration_count = array<int>::empty(queue, 1);

        const auto daal_initial_centroids =
            interop::convert_to_daal_sycl_homogen_table(queue,
                                                        arr_initial_centroids,
                                                        cluster_count,
                                                        column_count);
        const auto daal_centroids = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                                arr_centroids,
                                                                                cluster_count,
                                                                                column_count);
        const auto daal_labels =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_labels, row_count, 1);
        const auto daal_objective_function_value =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_objective_function_value, 1, 1);
        const auto daal_iteration_count =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_iteration_count, 1, 1);

        daal::data_management::NumericTable* daal_input[2] = { daal_data.get(),
                                                               daal_initial_centroids.get() };

        daal::data_management::NumericTable* daal_output[4] = { daal_centroids.get(),
                                                                daal_labels.get(),
                                                                daal_objective_function_value.get(),
                                                                daal_iteration_count.get() };

        interop::status_to_exception(
            daal_kmeans_lloyd_dense_ucapi_kernel_t<Float>().compute(daal_input, daal_output, &par));

        return train_result()
            .set_labels(
                dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build())
            .set_iteration_count(static_cast<std::int64_t>(arr_iteration_count[0]))
            .set_objective_function_value(static_cast<double>(arr_objective_function_value[0]))
            .set_model(model().set_centroids(dal::detail::homogen_table_builder{}
                                                 .reset(arr_centroids, cluster_count, column_count)
                                                 .build()));
    }
};

template struct train_kernel_gpu<float, method::lloyd_dense>;
template struct train_kernel_gpu<double, method::lloyd_dense>;

} // namespace oneapi::dal::kmeans::backend
