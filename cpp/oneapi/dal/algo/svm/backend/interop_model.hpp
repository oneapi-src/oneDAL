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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::svm::backend {

namespace interop  = dal::backend::interop;
namespace daal_svm = daal::algorithms::svm;

struct daal_model_builder : public daal::algorithms::svm::Model {
    daal_model_builder() = default;
    virtual ~daal_model_builder() {}

    auto& set_support_vectors(daal::data_management::NumericTablePtr support_vectors) {
        _SV = support_vectors;
        return *this;
    }

    auto& set_coefficients(daal::data_management::NumericTablePtr coefficients) {
        _SVCoeff = coefficients;
        return *this;
    }

    auto& set_bias(double bias) {
        _bias = bias;
        return *this;
    }
};

template <typename T>
inline auto convert_from_daal_model(daal_svm::Model& model) {
    auto table_support_vectors =
        interop::convert_from_daal_homogen_table<T>(model.getSupportVectors());
    auto table_classification_coefficients =
        interop::convert_from_daal_homogen_table<T>(model.getClassificationCoefficients());
    const double bias                        = model.getBias();
    const std::int64_t support_vectors_count = table_support_vectors.get_row_count();

    return dal::svm::model()
        .set_support_vectors(table_support_vectors)
        .set_coefficients(table_classification_coefficients)
        .set_bias(bias)
        .set_support_vector_count(support_vectors_count);
}

} // namespace oneapi::dal::svm::backend
