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

#pragma once

#include "algorithms/classifier/classifier_model.h"
#include "oneapi/dal/algo/knn/common.hpp"

namespace oneapi::dal::knn::backend {

namespace daal_cls = daal::algorithms::classifier;

class model_interop {
public:
    model_interop(const daal_cls::ModelPtr& daal_model) : daal_model(daal_model) {}
    void set_daal_model(const daal_cls::ModelPtr& model) {
        daal_model = model;
    }
    daal_cls::ModelPtr get_daal_model() {
        return daal_model;
    }

private:
    daal_cls::ModelPtr daal_model;
};

} // namespace oneapi::dal::knn::backend
