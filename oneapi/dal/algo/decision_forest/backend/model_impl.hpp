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
#include "oneapi/dal/algo/decision_tree/backend/node_info_impl.hpp"
#include "oneapi/dal/algo/decision_tree/backend/node_visitor_interop.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::decision_forest {

namespace dt = oneapi::dal::decision_tree;

template <typename Task>
struct daal_model_map;

template <>
struct daal_model_map<task::classification> {
    using daal_model_interop_t = backend::model_interop_cls;
};

template <>
struct daal_model_map<task::regression> {
    using daal_model_interop_t = backend::model_interop_reg;
};

#define DF_SERIALIZABLE(Task, ClassificationId, RegressionId)          \
    ONEDAL_SERIALIZABLE_MAP2(Task,                                     \
                             (task::classification, ClassificationId), \
                             (task::regression, RegressionId))

template <typename Task>
class detail::model_impl : public DF_SERIALIZABLE(Task,
                                                  decision_forest_classification_model_impl_id,
                                                  decision_forest_regression_model_impl_id) {
    static_assert(is_valid_task_v<Task>);

    using dtree_task_t = detail::decision_tree_task_map_t<Task>;
    using visitor_t = detail::decision_tree_visitor_ptr<Task>;
    using visitor_interop_t = decision_tree::visitor_interop<dtree_task_t>;

public:
    using task_t = Task;

    model_impl() = default;
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;

    explicit model_impl(backend::model_interop* interop) : interop_(interop) {
        ONEDAL_ASSERT(interop_);
    }

    virtual ~model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() const {
        return interop_;
    }

    void traverse_depth_first_impl(std::int64_t tree_idx, visitor_t&& visitor) const {
        if (!interop_) {
            throw dal::internal_error(
                dal::detail::error_messages::input_model_is_not_initialized());
        }

        auto daal_model =
            static_cast<const typename daal_model_map<Task>::daal_model_interop_t*>(interop_)
                ->get_model();
        if constexpr (std::is_same_v<Task, task::classification>) {
            visitor_interop_t vis(std::move(visitor), class_count);
            daal_model->traverseDFS(dal::detail::integral_cast<std::size_t>(tree_idx), vis);
        }
        else if constexpr (std::is_same_v<Task, task::regression>) {
            visitor_interop_t vis(std::move(visitor));
            daal_model->traverseDFS(dal::detail::integral_cast<std::size_t>(tree_idx), vis);
        }
    }

    void traverse_breadth_first_impl(std::int64_t tree_idx, visitor_t&& visitor) const {
        if (!interop_) {
            throw dal::internal_error(
                dal::detail::error_messages::input_model_is_not_initialized());
        }

        auto daal_model =
            static_cast<const typename daal_model_map<Task>::daal_model_interop_t*>(interop_)
                ->get_model();
        if constexpr (std::is_same_v<Task, task::classification>) {
            visitor_interop_t vis(std::move(visitor), class_count);
            daal_model->traverseBFS(dal::detail::integral_cast<std::size_t>(tree_idx), vis);
        }
        else if constexpr (std::is_same_v<Task, task::regression>) {
            visitor_interop_t vis(std::move(visitor));
            daal_model->traverseBFS(dal::detail::integral_cast<std::size_t>(tree_idx), vis);
        }
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(tree_count);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ar(class_count);
        }

        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(tree_count);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ar(class_count);
        }

        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

    std::int64_t tree_count = 0;
    std::int64_t class_count = 0;

private:
    backend::model_interop* interop_ = nullptr;
};

namespace backend {

using model_impl_cls = detail::model_impl<task::classification>;
using model_impl_reg = detail::model_impl<task::regression>;

} // namespace backend
} // namespace oneapi::dal::decision_forest
