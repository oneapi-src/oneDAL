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
struct binary_label_t {
    Float first;
    Float second;

    bool operator==(const binary_label_t&& binary_label) const {
        return (first == binary_label.first && second == binary_label.second) ||
               (first == binary_label.second && second == binary_label.first);
    }
};

template <typename Float>
inline binary_label_t<Float> get_unique_labels_impl(const array<Float>& arr_label) {
    const std::int64_t count = arr_label.get_count();

    Float first_label = arr_label[0];
    Float second_label = arr_label[1];

    for (std::int64_t i = 1; i < count; ++i) {
        if (arr_label[i] != first_label) {
            second_label = arr_label[i];
            break;
        }
    }

    if (first_label == second_label) {
        throw invalid_argument(
            dal::detail::error_messages::input_labels_contain_only_one_unique_value_expect_two());
    }

    // need to sort class labels to be consistent with Scikit-learn*
    if (first_label > second_label) {
        std::swap(first_label, second_label);
    }

    return { first_label, second_label };
}

template <typename Float>
inline void convert_binary_labels_impl(const binary_label_t<Float>& requested_unique_labels,
                                       const binary_label_t<Float>& old_unique_labels,
                                       const array<Float>& arr_label,
                                       array<Float>& new_label_arr) {
    const std::int64_t count = arr_label.get_count();
    auto new_label_data = new_label_arr.get_mutable_data();

    for (std::int64_t i = 0; i < count; ++i) {
        if (arr_label[i] == old_unique_labels.first) {
            new_label_data[i] = requested_unique_labels.first;
        }
        else if (arr_label[i] == old_unique_labels.second) {
            new_label_data[i] = requested_unique_labels.second;
        }
        else {
            throw invalid_argument(dal::detail::error_messages::
                                       input_labels_contain_wrong_unique_values_count_expect_two());
        }
    }
}

template <typename Float>
inline binary_label_t<Float> get_unique_labels(const table& labels) {
    auto arr_label = row_accessor<const Float>{ labels }.pull();
    return get_unique_labels_impl<Float>(arr_label);
}

template <typename Float>
inline table convert_binary_labels(const table& labels,
                                   const binary_label_t<Float>& requested_unique_labels,
                                   const binary_label_t<Float>& old_unique_labels) {
    if (old_unique_labels == binary_label_t<Float>{ 0, 1 } ||
        old_unique_labels == binary_label_t<Float>{ -1, 1 }) {
        return labels;
    }
    else {
        ONEDAL_ASSERT(labels.get_column_count() == 1);

        auto arr_label = row_accessor<const Float>{ labels }.pull();
        const std::int64_t count = arr_label.get_count();

        auto new_label_arr = array<Float>::empty(count);
        convert_binary_labels_impl<Float>(requested_unique_labels,
                                          old_unique_labels,
                                          arr_label,
                                          new_label_arr);

        return homogen_table::wrap(new_label_arr, count, 1);
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float>
inline binary_label_t<Float> get_unique_labels(sycl::queue& queue, const table& labels) {
    auto arr_label = row_accessor<const Float>{ labels }.pull(queue);
    const auto arr_label_host = dal::backend::to_host_sync(arr_label);
    return get_unique_labels_impl<Float>(arr_label_host);
}

template <typename Float>
inline table convert_binary_labels(sycl::queue& queue,
                                   const table& labels,
                                   const binary_label_t<Float>& requested_unique_labels,
                                   const binary_label_t<Float>& old_unique_labels) {
    if (old_unique_labels == binary_label_t<Float>{ 0, 1 } ||
        old_unique_labels == binary_label_t<Float>{ -1, 1 }) {
        return labels;
    }
    else {
        ONEDAL_ASSERT(labels.get_column_count() == 1);

        auto arr_label = row_accessor<const Float>{ labels }.pull(queue);
        const std::int64_t count = arr_label.get_count();

        auto new_label_arr = array<Float>::empty(queue, count, sycl::usm::alloc::host);
        convert_binary_labels_impl<Float>(requested_unique_labels,
                                          old_unique_labels,
                                          arr_label,
                                          new_label_arr);

        auto device_arr_data =
            dal::backend::to_device_sync(queue, new_label_arr).get_mutable_data();
        return homogen_table::wrap(queue, device_arr_data, count, 1);
    }
}
#endif

} // namespace oneapi::dal::svm::backend
