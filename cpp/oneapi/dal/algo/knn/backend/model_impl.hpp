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

#include "oneapi/dal/algo/knn/common.hpp"
#include "oneapi/dal/algo/knn/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h>
#else
#include <daal/src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h>
#endif

namespace oneapi::dal::knn {

#define KNN_SERIALIZABLE(Task, ClassificationId, SearchId)             \
    ONEDAL_SERIALIZABLE_MAP2(Task,                                     \
                             (task::classification, ClassificationId), \
                             (task::search, SearchId))

template <typename Task>
class detail::v1::model_impl : public base {
public:
    model_impl() = default;

    virtual backend::model_interop* get_interop() = 0;
};

namespace backend {

template <typename Task>
using model_impl = detail::model_impl<Task>;

template <typename Task>
class brute_force_model_impl : public model_impl<Task>,
                               public KNN_SERIALIZABLE(Task,
                                                       knn_brute_force_classification_model_impl_id,
                                                       knn_brute_force_search_model_impl_id) {
public:
    brute_force_model_impl() = default;

    brute_force_model_impl(const table& data, const table& responses)
            : data_(data),
              responses_(responses) {}

    backend::model_interop* get_interop() override {
        return nullptr;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(data_, responses_);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(data_, responses_);
    }

    table get_data() {
        return data_;
    }

    table get_responses() {
        return responses_;
    }

private:
    table data_;
    table responses_;
};

template <typename Task>
class kd_tree_model_impl : public model_impl<Task>,
                           public KNN_SERIALIZABLE(Task,
                                                   knn_kd_tree_classification_model_impl_id,
                                                   knn_kd_tree_search_model_impl_id) {
public:
    kd_tree_model_impl() : interop_(nullptr) {}
    kd_tree_model_impl(const kd_tree_model_impl&) = delete;
    kd_tree_model_impl& operator=(const kd_tree_model_impl&) = delete;

    kd_tree_model_impl(backend::model_interop* interop) : interop_(interop) {}

    ~kd_tree_model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() override {
        return interop_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

private:
    backend::model_interop* interop_;
};

template <typename Task>
inline auto dynamic_cast_to_bf_knn_model(const model<Task>& m) {
    const auto trained_model =
        dynamic_cast<oneapi::dal::knn::backend::brute_force_model_impl<Task>*>(
            &dal::detail::get_impl(m));

    if (!trained_model) {
        throw invalid_argument{ dal::detail::error_messages::incompatible_knn_model() };
    }

    return trained_model;
}

template <typename Float>
inline auto create_daal_model_for_bf_knn(
    const daal::data_management::NumericTablePtr daal_train_data,
    const daal::data_management::NumericTablePtr daal_train_responses) {
    namespace daal_bf_knn = daal::algorithms::bf_knn_classification;
    const std::int64_t column_count = daal_train_data->getNumberOfColumns();

    const auto model_ptr = daal_bf_knn::ModelPtr(new daal_bf_knn::Model(column_count));

    // Data or responses should not be copied
    model_ptr->impl()->setData<Float>(daal_train_data, false);
    model_ptr->impl()->setLabels<Float>(daal_train_responses, false);

    return model_ptr;
}

template <typename Float, typename Task>
inline auto convert_onedal_to_daal_knn_model(const model<Task>& m) {
    namespace interop = dal::backend::interop;

    const auto trained_model = dynamic_cast_to_bf_knn_model(m);

    const auto daal_train_data = interop::convert_to_daal_table<Float>(trained_model->get_data());
    const auto daal_train_responses =
        interop::convert_to_daal_table<Float>(trained_model->get_responses());

    const auto model_ptr =
        create_daal_model_for_bf_knn<Float>(daal_train_data, daal_train_responses);

    return model_ptr;
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float, typename Task>
inline auto convert_onedal_to_daal_knn_model(const sycl::queue& queue, const model<Task>& m) {
    namespace interop = dal::backend::interop;

    const auto trained_model = dynamic_cast_to_bf_knn_model<Task>(m);

    const auto daal_train_data = interop::convert_to_daal_table(queue, trained_model->get_data());
    const auto daal_train_responses =
        interop::convert_to_daal_table(queue, trained_model->get_responses());

    const auto model_ptr =
        create_daal_model_for_bf_knn<Float>(daal_train_data, daal_train_responses);

    return model_ptr;
}
#endif

} // namespace backend
} // namespace oneapi::dal::knn
