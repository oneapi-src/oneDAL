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

#pragma once

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include <daal/include/algorithms/svm/svm_model.h>
#include <daal/include/algorithms/multi_class_classifier/multi_class_classifier_model.h>

namespace oneapi::dal::svm::backend {

namespace interop = dal::backend::interop;
namespace daal_svm = daal::algorithms::svm;
namespace daal_multiclass = daal::algorithms::multi_class_classifier;

struct daal_model_builder : public daal::algorithms::svm::Model {
    daal_model_builder() = default;
    virtual ~daal_model_builder() {}

    auto& set_support_vectors(daal::data_management::NumericTablePtr support_vectors) {
        _SV = support_vectors;
        return *this;
    }

    auto& set_coeffs(daal::data_management::NumericTablePtr coeffs) {
        _SVCoeff = coeffs;
        return *this;
    }

    auto& set_bias(double bias) {
        _bias = bias;
        return *this;
    }
};

class model_interop : public base {
public:
    virtual ~model_interop() = default;
    virtual void clear() {}
};

template <typename DaalModel>
class model_interop_impl : public model_interop {
public:
    model_interop_impl(DaalModel& model) : daal_model_(model) {}

    const DaalModel get_model() const {
        return daal_model_;
    }

    void clear() override {
        // daal_model_->clear();
    }

private:
    DaalModel daal_model_;
};

using model_interop_cls = model_interop_impl<daal_multiclass::ModelPtr>;

template <typename Task, typename Float>
inline auto convert_from_daal_model(daal_svm::Model& model) {
    auto table_support_vectors =
        interop::convert_from_daal_homogen_table<Float>(model.getSupportVectors());
    auto table_classification_coeffs =
        interop::convert_from_daal_homogen_table<Float>(model.getClassificationCoefficients());
    const double bias = model.getBias();
    auto arr_biases = array<Float>::full(1, static_cast<Float>(bias));

    return dal::svm::model<Task>()
        .set_support_vectors(table_support_vectors)
        .set_coeffs(table_classification_coeffs)
        .set_bias(bias)
        .set_biases(dal::detail::homogen_table_builder{}.reset(arr_biases, 1, 1).build());
}

template <typename Task, typename Float>
inline auto convert_from_daal_multiclass_model(daal_multiclass::Model& model) {
    std::int64_t model_count = model.getNumberOfTwoClassClassifierModels();
    printf("model_count: %lu \n", model_count);
    for (std::int64_t i = 0; i < model_count; ++i) {
        auto svm_model =
            daal::services::staticPointerCast<daal_svm::Model>(model.getTwoClassClassifierModel(i));
    }

    return dal::svm::model<Task>();

    //     auto table_support_vectors =
    //         interop::convert_from_daal_homogen_table<Float>(model.getSupportVectors());
    // auto table_classification_coeffs =
    //     interop::convert_from_daal_homogen_table<Float>(model.getClassificationCoefficients());
    // const double bias = model.getBias();

    // return dal::svm::model<Task>()
    //     .set_support_vectors(table_support_vectors)
    //     .set_coeffs(table_classification_coeffs)
    //     .set_bias(bias);
}

} // namespace oneapi::dal::svm::backend
