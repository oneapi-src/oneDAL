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

#pragma once

#include "oneapi/dal/algo/svm/common.hpp"
#include "oneapi/dal/algo/svm/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::svm {

#define SVM_SERIALIZABLE(Task, ClassificationId, RegressionId)         \
    ONEDAL_SERIALIZABLE_MAP2(Task,                                     \
                             (task::classification, ClassificationId), \
                             (task::regression, RegressionId))

template <typename Task>
class detail::v1::model_impl : public SVM_SERIALIZABLE(Task,
                                                       svm_classification_model_impl_id,
                                                       svm_regression_model_impl_id) {
public:
    model_impl() = default;
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;
    model_impl(backend::model_interop* interop) : interop_(interop) {}

    table support_vectors;
    table coeffs;
    double bias;
    table biases;
    double first_class_label;
    double second_class_label;
    std::int64_t class_count = 2;

    backend::model_interop* get_interop() const {
        return interop_;
    }

    bool has_interop() const {
        return interop_ != nullptr;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(support_vectors, coeffs, bias, biases);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ar(first_class_label, second_class_label, class_count);
        }

        // TODO:
        // ar(has_interop());
        // if (has_interop()) {
        //     dal::detail::serialize_polymorphic(interop_, ar);
        // }
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(support_vectors, coeffs, bias, biases);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ar(first_class_label, second_class_label, class_count);
        }

        // TODO:
        // const bool has_interop_in_archive = ar.pop<bool>();
        // if (has_interop_in_archive) {
        //     interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
        // }
    }

private:
    backend::model_interop* interop_ = nullptr;
};

namespace backend {

using model_impl_cls = detail::model_impl<task::classification>;
using model_impl_reg = detail::model_impl<task::regression>;

} // namespace backend
} // namespace oneapi::dal::svm
