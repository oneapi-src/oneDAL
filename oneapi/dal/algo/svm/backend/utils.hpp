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

#include <daal/include/algorithms/svm/svm_train_types.h>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::svm::backend {

namespace daal_svm = daal::algorithms::svm;

template <daal_svm::training::Method Value>
using daal_method_constant = std::integral_constant<daal_svm::training::Method, Value>;

template <typename Method>
struct to_daal_method;

template <>
struct to_daal_method<method::smo> : daal_method_constant<daal_svm::training::boser> {};

template <>
struct to_daal_method<method::thunder> : daal_method_constant<daal_svm::training::thunder> {};

template <typename Float>
struct binary_response_t {
    Float first;
    Float second;

    bool operator==(const binary_response_t&& binary_response) const {
        return (first == binary_response.first && second == binary_response.second) ||
               (first == binary_response.second && second == binary_response.first);
    }
};

template <typename Float>
inline binary_response_t<Float> get_unique_responses_impl(const array<Float>& arr_response) {
    const std::int64_t count = arr_response.get_count();

    Float first_response = arr_response[0];
    Float second_response = arr_response[1];

    for (std::int64_t i = 1; i < count; ++i) {
        if (arr_response[i] != first_response) {
            second_response = arr_response[i];
            break;
        }
    }

    if (first_response == second_response) {
        throw invalid_argument(dal::detail::error_messages::
                                   input_responses_contain_only_one_unique_value_expect_two());
    }

    // need to sort class responses to be consistent with Scikit-learn*
    if (first_response > second_response) {
        std::swap(first_response, second_response);
    }

    return { first_response, second_response };
}

template <typename Float>
inline void convert_binary_responses_impl(
    const binary_response_t<Float>& requested_unique_responses,
    const binary_response_t<Float>& old_unique_responses,
    const array<Float>& arr_response,
    array<Float>& new_response_arr) {
    const std::int64_t count = arr_response.get_count();
    auto new_response_data = new_response_arr.get_mutable_data();

    for (std::int64_t i = 0; i < count; ++i) {
        if (arr_response[i] == old_unique_responses.first) {
            new_response_data[i] = requested_unique_responses.first;
        }
        else if (arr_response[i] == old_unique_responses.second) {
            new_response_data[i] = requested_unique_responses.second;
        }
        else {
            throw invalid_argument(
                dal::detail::error_messages::
                    input_responses_contain_wrong_unique_values_count_expect_two());
        }
    }
}

template <typename Float>
inline binary_response_t<Float> get_unique_responses(const table& responses) {
    auto arr_response = row_accessor<const Float>{ responses }.pull();
    return get_unique_responses_impl<Float>(arr_response);
}

template <typename Float>
inline table convert_binary_responses(const table& responses,
                                      const binary_response_t<Float>& requested_unique_responses,
                                      const binary_response_t<Float>& old_unique_responses) {
    if (old_unique_responses == binary_response_t<Float>{ 0, 1 } ||
        old_unique_responses == binary_response_t<Float>{ -1, 1 }) {
        return responses;
    }
    else {
        ONEDAL_ASSERT(responses.get_column_count() == 1);

        auto arr_response = row_accessor<const Float>{ responses }.pull();
        const std::int64_t count = arr_response.get_count();

        auto new_response_arr = array<Float>::empty(count);
        convert_binary_responses_impl<Float>(requested_unique_responses,
                                             old_unique_responses,
                                             arr_response,
                                             new_response_arr);

        return homogen_table::wrap(new_response_arr, count, 1);
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float>
inline binary_response_t<Float> get_unique_responses(sycl::queue& queue, const table& responses) {
    auto arr_response = row_accessor<const Float>{ responses }.pull(queue);
    const auto arr_response_host = dal::backend::to_host_sync(arr_response);
    return get_unique_responses_impl<Float>(arr_response_host);
}

template <typename Float>
inline table convert_binary_responses(sycl::queue& queue,
                                      const table& responses,
                                      const binary_response_t<Float>& requested_unique_responses,
                                      const binary_response_t<Float>& old_unique_responses) {
    if (old_unique_responses == binary_response_t<Float>{ -1, 1 }) {
        return responses;
    }
    else {
        ONEDAL_ASSERT(responses.get_column_count() == 1);

        auto arr_response = row_accessor<const Float>{ responses }.pull(queue);
        const std::int64_t count = arr_response.get_count();

        const auto arr_response_host = dal::backend::to_host_sync(arr_response);
        auto new_response_arr = array<Float>::empty(queue, count, sycl::usm::alloc::host);
        convert_binary_responses_impl<Float>(requested_unique_responses,
                                             old_unique_responses,
                                             arr_response_host,
                                             new_response_arr);

        auto device_arr_data = dal::backend::to_device_sync(queue, new_response_arr);
        return homogen_table::wrap(device_arr_data, count, 1);
    }
}
#endif

} // namespace oneapi::dal::svm::backend
