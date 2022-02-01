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

#include <daal/include/algorithms/svm/svm_model.h>
#include <daal/include/algorithms/multi_class_classifier/multi_class_classifier_model.h>
#include <daal/src/algorithms/multiclassclassifier/multiclassclassifier_svm_model.h>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/algo/svm/backend/model_impl.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::svm::backend {

namespace interop = dal::backend::interop;
namespace daal_svm = daal::algorithms::svm;
namespace daal_multiclass_internal = daal_multiclass::internal;

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

template <typename Task, typename Float>
inline auto convert_from_daal_model(daal_svm::Model& daal_model) {
    auto table_support_vectors =
        interop::convert_from_daal_table<Float>(daal_model.getSupportVectors());
    auto table_classification_coeffs =
        interop::convert_from_daal_homogen_table<Float>(daal_model.getClassificationCoefficients());
    const double bias = daal_model.getBias();
    auto arr_biases = array<Float>::full(1, static_cast<Float>(bias));

    auto model =
        dal::svm::model<Task>()
            .set_support_vectors(table_support_vectors)
            .set_coeffs(table_classification_coeffs)
            .set_biases(dal::detail::homogen_table_builder{}.reset(arr_biases, 1, 1).build());

    dal::detail::get_impl(model).bias = bias;
    return model;
}

template <typename T>
inline array<T> convert_from_daal_table_to_array(const daal::data_management::NumericTablePtr& nt) {
    daal::data_management::BlockDescriptor<T> block;
    const std::int64_t row_count = nt->getNumberOfRows();
    const std::int64_t column_count = nt->getNumberOfColumns();

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    T* data = block.getBlockPtr();
    array<T> arr(data, row_count * column_count, [nt, block](T* p) mutable {
        nt->releaseBlockOfRows(block);
    });
    return arr;
}

template <typename Task, typename Float>
inline auto convert_from_daal_multiclass_model(
    const daal_multiclass_internal::SvmModelPtr& daal_model) {
    auto table_biases = interop::convert_from_daal_homogen_table<Float>(daal_model->getBiases());
    auto table_coeffs =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getCoefficients());
    auto table_support_vectors =
        interop::convert_from_daal_table<Float>(daal_model->getSupportVectors());

    return dal::svm::model<Task>()
        .set_support_vectors(table_support_vectors)
        .set_coeffs(table_coeffs)
        .set_biases(table_biases);
}

} // namespace oneapi::dal::svm::backend
