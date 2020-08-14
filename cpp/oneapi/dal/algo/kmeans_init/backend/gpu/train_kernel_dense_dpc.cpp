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

#include <src/algorithms/kmeans/oneapi/kmeans_init_dense_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/kmeans_init/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using std::int64_t;
using dal::backend::context_gpu;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop          = dal::backend::interop;

template <typename Float>
using daal_kmeans_init_dense_batch_ucapi_kernel_t =
    daal_kmeans_init::internal::KMeansInitDenseBatchKernelUCAPI<daal_kmeans_init::defaultDense,
                                                                Float>;

template <typename Float>
struct train_kernel_gpu<Float, method::dense> {
    train_result operator()(const dal::backend::context_gpu& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        auto& queue = ctx.get_queue();
        interop::execution_context_guard guard(queue);

        const auto data = input.get_data();

        const int64_t column_count  = data.get_column_count();
        const int64_t cluster_count = params.get_cluster_count();

        daal_kmeans_init::Parameter par(cluster_count);

        auto arr_data        = row_accessor<const Float>{ data }.pull(queue);
        const auto daal_data = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                           arr_data,
                                                                           data.get_row_count(),
                                                                           data.get_column_count());

        array<Float> arr_centroids = array<Float>::empty(queue, cluster_count * column_count);
        const auto daal_centroids  = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                                arr_centroids,
                                                                                cluster_count,
                                                                                column_count);

        const size_t len_daal_input                                       = 1;
        daal::data_management::NumericTable* daal_input[len_daal_input]   = { daal_data.get() };
        const size_t len_daal_output                                      = 1;
        daal::data_management::NumericTable* daal_output[len_daal_output] = {
            daal_centroids.get()
        };

        interop::status_to_exception(
            daal_kmeans_init_dense_batch_ucapi_kernel_t<Float>().compute(len_daal_input,
                                                                         daal_input,
                                                                         len_daal_output,
                                                                         daal_output,
                                                                         &par,
                                                                         *(par.engine)));

        return train_result().set_centroids(dal::detail::homogen_table_builder{}
                                                .reset(arr_centroids, cluster_count, column_count)
                                                .build());
    }
};

template struct train_kernel_gpu<float, method::dense>;
template struct train_kernel_gpu<double, method::dense>;

} // namespace oneapi::dal::kmeans_init::backend
