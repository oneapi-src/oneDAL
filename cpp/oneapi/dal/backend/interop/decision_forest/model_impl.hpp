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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/detail/model_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

#include <daal/include/algorithms/decision_forest/decision_forest_classification_model.h>

namespace oneapi::dal::backend::interop::decision_forest {

namespace daal_df = daal::algorithms::decision_forest;

namespace df = oneapi::dal::decision_forest;

/* oneDal -> daal model bridge */
class interop_model_impl : public df::detail::model_impl {
public:
    interop_model_impl() = delete;
    interop_model_impl(daal_df::classification::ModelPtr pmdl) : pmdl_(pmdl){};
    virtual ~interop_model_impl(){};

    virtual std::int64_t get_tree_count() const {
        return pmdl_->numberOfTrees();
    }
    virtual std::int64_t get_class_count() const {
        return pmdl_->getNumberOfClasses();
    }
    void clear() {
        pmdl_->clear();
    }

    virtual bool is_interop() const {
        return true;
    }

    void set_model(daal_df::classification::ModelPtr pmdl) {
        pmdl_ = pmdl;
    }
    daal_df::classification::ModelPtr get_model() {
        return pmdl_;
    }

private:
    daal_df::classification::ModelPtr pmdl_;
};

} // namespace oneapi::dal::backend::interop::decision_forest
