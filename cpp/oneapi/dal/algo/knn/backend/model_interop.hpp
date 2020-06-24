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

#include "oneapi/dal/algo/knn/common.hpp"

namespace oneapi::dal::knn {

namespace daal_knn = daal::algorithms::kdtree_knn_classification;
using daal_knn_classification_model_t = daal_knn::training::internal::KNNClassificationTrainBatchKernel<daal::batch, Float, Cpu>;


class interop_model {
public:
    interop_model() : daal_model_(nullptr) {}
    void set_daal_model(daal_knn_classification_model_t * model) { daal_model_ = model;}
    daal_knn_classification_model_t * get_daal_model() {return daal_model_;}
private:
    daal_knn_classification_model_t * daal_model_;
};

} // namespace oneapi::dal::knn