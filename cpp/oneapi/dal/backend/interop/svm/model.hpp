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

namespace oneapi::dal::backend::interop::svm {

struct daal_model : public daal::algorithms::svm::Model {
    daal_model() = default;
    virtual ~daal_model() {}

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

} // namespace oneapi::dal::backend::interop::svm
