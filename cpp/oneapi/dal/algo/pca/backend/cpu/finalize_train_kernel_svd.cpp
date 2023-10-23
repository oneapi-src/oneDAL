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

#include <daal/src/algorithms/pca/pca_dense_svd_online_kernel.h>

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/finalize_train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::backend {

using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;
using model_t = model<task::dim_reduction>;

namespace interop = dal::backend::interop;

namespace daal_pca = daal::algorithms::pca;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_svd_kernel_t = daal_pca::internal::PCASVDOnlineKernel<Float, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel_finalize_train(const context_cpu& ctx,
                                                          const descriptor_t& desc,
                                                          const partial_train_result<Task>& input) {
    const std::int64_t component_count =
        get_component_count(desc, input.get_partial_crossproduct());
    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();

    auto result = train_result<task::dim_reduction>{}.set_result_options(desc.get_result_options());
    daal::services::SharedPtr<DataCollection> DataCollectionPtr;
    auto arr_eigvec = array<Float>::empty(column_count * column_count);
    auto arr_eigval = array<Float>::empty(1 * component_count);

    const auto daal_eigenvectors =
        interop::convert_to_daal_homogen_table(arr_eigvec, column_count, column_count);
    const auto daal_eigenvalues =
        interop::convert_to_daal_homogen_table(arr_eigval, 1, component_count);

    const auto daal_nobs = interop::convert_to_daal_table<Float>(input.get_partial_n_rows());
    daal::data_management::DataCollectionPtr decomposeCollection =
        daal::data_management::DataCollectionPtr(new daal::data_management::DataCollection());

    for (std::int64_t i = 0; i < input.get_auxialry_table_count(); i++) {
        const auto daal_crossproduct =
            interop::copy_to_daal_homogen_table<Float>(input.get_auxialry_table(i));
        decomposeCollection->push_back(daal_crossproduct);
    }
    daal_pca::internal::InputDataType dtype = daal_pca::internal::nonNormalizedDataset;
    interop::status_to_exception(
        interop::call_daal_kernel_finalize_merge<Float, daal_svd_kernel_t>(ctx,
                                                                           dtype,
                                                                           daal_nobs,
                                                                           *daal_eigenvalues,
                                                                           *daal_eigenvectors,
                                                                           decomposeCollection));

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        auto reshaped_array = arr_eigvec.get_slice(0, component_count * column_count);
        const auto mdl = model_t{}.set_eigenvectors(
            homogen_table::wrap(reshaped_array, component_count, column_count));
        result.set_model(mdl);
    }

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(arr_eigval, 1, component_count));
    }

    return result;
}

template <typename Float, typename Task>
static train_result<Task> finalize_train(const context_cpu& ctx,
                                         const descriptor_t& desc,
                                         const partial_train_result<Task>& input) {
    return call_daal_kernel_finalize_train<Float>(ctx, desc, input);
}

template <typename Float>
struct finalize_train_kernel_cpu<Float, method::svd, task::dim_reduction> {
    train_result<task::dim_reduction> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_train_result<task::dim_reduction>& input) const {
        return finalize_train<Float, task::dim_reduction>(ctx, desc, input);
    }
};

template struct finalize_train_kernel_cpu<float, method::svd, task::dim_reduction>;
template struct finalize_train_kernel_cpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
