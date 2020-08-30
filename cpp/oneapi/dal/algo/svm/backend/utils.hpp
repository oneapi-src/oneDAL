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

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::svm::backend {

template <typename Float>
struct binary_label_t {
    Float first;
    Float second;
};

template <typename Float>
static array<Float> convert_labels(const array<Float>& arr_label,
                                   const binary_label_t<Float>& in_binary_labels,
                                   binary_label_t<Float>& out_binary_labels) {
    const std::int64_t count = arr_label.get_count();
    auto new_label_arr = array<Float>::empty(count);
    auto new_label_data = new_label_arr.get_mutable_data();

    Float value_first_class_label = arr_label[0];
    Float value_second_class_label = arr_label[1];

    new_label_data[0] = in_binary_labels.first;
    std::int64_t i = 1;
    for (; i < count; ++i) {
        if (arr_label[i] == value_first_class_label) {
            new_label_data[i] = in_binary_labels.first;
        }
        else {
            value_second_class_label = arr_label[i];
            new_label_data[i] = in_binary_labels.second;
            break;
        }
    }
    if (value_first_class_label == value_second_class_label) {
        throw invalid_argument("Input label data should have more than one unique label");
    }

    for (; i < count; ++i) {
        if (arr_label[i] == value_first_class_label) {
            new_label_data[i] = in_binary_labels.first;
        }
        else if (arr_label[i] == value_second_class_label) {
            new_label_data[i] = in_binary_labels.second;
        }
        else {
            throw invalid_argument("Input label data should have only two unique labels");
        }
    }

    out_binary_labels.first = value_first_class_label;
    out_binary_labels.second = value_second_class_label;
    return new_label_arr;
}

#ifdef ONEAPI_DAL_DATA_PARALLEL

template <typename Float>
static array<Float> convert_labels(const sycl::queue& queue,
                                   const array<Float>& arr_label,
                                   const binary_label_t<Float>& in_binary_labels,
                                   binary_label_t<Float>& out_binary_labels) {
    // TODO: make for dpcpp kernel
    const std::int64_t count = arr_label.get_count();
    auto new_label_arr = array<Float>::empty(queue, count);
    auto new_label_data = new_label_arr.get_mutable_data();

    Float value_first_class_label = arr_label[0];
    Float value_second_class_label = arr_label[1];

    new_label_data[0] = in_binary_labels.first;
    std::int64_t i = 1;
    for (; i < count; ++i) {
        if (arr_label[i] == value_first_class_label) {
            new_label_data[i] = in_binary_labels.first;
        }
        else {
            value_second_class_label = arr_label[i];
            new_label_data[i] = in_binary_labels.second;
            break;
        }
    }
    if (value_first_class_label == value_second_class_label) {
        throw invalid_argument("Input label data should have more one unique label");
    }

    for (; i < count; ++i) {
        if (arr_label[i] == value_first_class_label) {
            new_label_data[i] = in_binary_labels.first;
        }
        else if (arr_label[i] == value_second_class_label) {
            new_label_data[i] = in_binary_labels.second;
        }
        else {
            throw invalid_argument("Input label data should have only two unique labels");
        }
    }

    out_binary_labels.first = value_first_class_label;
    out_binary_labels.second = value_second_class_label;

    return new_label_arr;
}

#endif

} // namespace oneapi::dal::svm::backend
