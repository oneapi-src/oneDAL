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
};

template <typename DaalModel>
class model_interop_impl : public model_interop {
public:
    model_interop_impl(DaalModel& model) : daal_model_(model) {}

    const DaalModel get_model() const {
        return daal_model_;
    }

private:
    DaalModel daal_model_;
};

using model_interop_cls = model_interop_impl<daal_multiclass::ModelPtr>;

template <typename Task, typename Float>
inline auto convert_from_daal_model(daal_svm::Model& daal_model) {
    auto table_support_vectors =
        interop::convert_from_daal_homogen_table<Float>(daal_model.getSupportVectors());
    auto table_classification_coeffs =
        interop::convert_from_daal_homogen_table<Float>(daal_model.getClassificationCoefficients());
    const double bias = daal_model.getBias();
    auto arr_biases = array<Float>::full(1, static_cast<Float>(bias));

    auto model =
        dal::svm::model<Task>()
            .set_support_vectors(table_support_vectors)
            .set_coeffs(table_classification_coeffs)
            .set_biases(dal::detail::homogen_table_builder{}.reset(arr_biases, 1, 1).build());

    // auto model_impl = dal::detail::pimpl_accessor().get_pimpl(model);
    // auto model_impl = dal::detail::get_impl(model);
    // model_impl->bias = bias;
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
inline auto convert_from_daal_multiclass_model(const daal_multiclass::ModelPtr& daal_model,
                                               const table& labels,
                                               const std::int64_t class_count) {
    const std::int64_t model_count = daal_model->getNumberOfTwoClassClassifierModels();
    const int64_t row_count = labels.get_row_count();
    auto arr_label = row_accessor<const Float>{ labels }.pull();

    auto arr_biases = array<Float>::empty(model_count);
    auto arr_label_indexes = array<std::int64_t>::empty(row_count);

    // auto arr_is_sv = array<bool>::full(row_count, false)
    // auto arr_coefs = array<Float>::empty(model_count * );

    auto data_biases = arr_biases.get_mutable_data();
    auto data_label_indexes = arr_label_indexes.get_mutable_data();
    // auto data_is_sv = arr_is_sv.get_mutable_data();
    // auto data_label_indexes = arr_biases.get_mutable_data();
    // auto arr_coeffs = array<Float>::empty(class_count - 1, );
    auto arr_offets = array<std::int64_t>::empty(model_count + 1);
    auto data_offets = arr_offets.get_mutable_data();
    data_offets[0] = 0;

    printf("model_count: %lu \n", model_count);
    std::int64_t i_model = 0;
    std::int64_t n_cv_count = 0;
    for (std::int64_t first_class = 0; first_class < class_count; ++first_class) {
        for (std::int64_t two_class = 0; two_class < first_class; ++two_class) {
            printf("%lu %lu\n", first_class, two_class);

            auto svm_model = daal::services::staticPointerCast<daal_svm::Model>(
                daal_model->getTwoClassClassifierModel(i_model));
            auto two_class_sv_ind = svm_model->getSupportIndices();
            auto arr_two_class_sv_ind = convert_from_daal_table_to_array<int>(two_class_sv_ind);

            const std::int64_t count_sv_ind = two_class_sv_ind->getNumberOfRows();
            for (std::int64_t i = 0; i < row_count; ++i) {
                if (std::int64_t(arr_label[i]) == first_class ||
                    std::int64_t(arr_label[i]) == two_class) {
                    for (std::int64_t j = 0; j < count_sv_ind; ++j) {
                        if (arr_two_class_sv_ind[j] == i) {
                            data_label_indexes[n_cv_count] = i;
                            ++n_cv_count;
                        }
                    }
                }
            }
            data_offets[i_model + 1] = n_cv_count;
            data_biases[i_model] = -svm_model->getBias();

            auto coefs = svm_model->getClassificationCoefficients();
            printf("n_coeffs: %lu; two_class_sv_ind: %lu\n",
                   coefs->getNumberOfRows(),
                   two_class_sv_ind->getNumberOfRows());
            ++i_model;
        }
    }

    auto arr_group_indices_by_class = array<std::int64_t>::full(n_cv_count, std::int64_t(0));
    auto arr_sv_ind_counters = array<std::int64_t>::full(class_count, std::int64_t(0));
    auto arr_sv_ind_counters_2 = array<std::int64_t>::full(class_count, std::int64_t(0));
    auto data_group_indices_by_class = arr_group_indices_by_class.get_mutable_data();
    auto data_sv_ind_counters = arr_sv_ind_counters.get_mutable_data();
    auto data_sv_ind_counters_2 = arr_sv_ind_counters.get_mutable_data();
    for (std::int64_t i = 0; i < n_cv_count; ++i) {
        ++data_sv_ind_counters[std::int64_t(arr_label[data_label_indexes[i]])];
    }
    for (std::int64_t i = 0; i < class_count; ++i) {
        printf("%lu ", data_sv_ind_counters[i]);
    }
    printf("\n");
    std::int64_t idx = 0;
    for (std::int64_t i = 0; i < class_count; ++i) {
        for (std::int64_t j = 0; j < n_cv_count; ++j) {
            if (std::int64_t(arr_label[data_label_indexes[j]]) == i) {
                data_group_indices_by_class[idx] = data_label_indexes[j];
                printf("%lu ", data_group_indices_by_class[idx]);
                idx++;
            }
        }
    }
    printf("\n");
    for (std::int64_t i = 0; i < n_cv_count; ++i) {
        std::int64_t sv_label = std::int64_t(arr_label[data_label_indexes[i]]);

        data_sv_ind_counters_2[sv_label]++;
    }

    auto arr_dual_coef = array<Float>::full((class_count - 1) * n_cv_count, std::int64_t(0));
    auto data_dual_coef = arr_dual_coef.get_mutable_data();
    i_model = 0;
    for (std::int64_t first_class = 0; first_class < class_count; ++first_class) {
        for (std::int64_t two_class = first_class + 1; two_class < first_class; ++two_class) {
            auto svm_model = daal::services::staticPointerCast<daal_svm::Model>(
                daal_model->getTwoClassClassifierModel(i_model));
            auto two_class_sv_ind = svm_model->getSupportIndices();
            auto coefs = svm_model->getClassificationCoefficients();

            auto arr_two_class_sv_ind = convert_from_daal_table_to_array<Float>(two_class_sv_ind);
            auto arr_coefs = convert_from_daal_table_to_array<Float>(coefs);

            for (std::int64_t k = data_offets[i_model]; k < data_offets[i_model + 1]; ++k) {
                const std::int64_t ind = data_label_indexes[k];
                const std::int64_t label = std::int64_t(arr_label[ind]);

                const std::int64_t row_index = label == two_class ? first_class : two_class - 1;
                data_dual_coef[row_index * n_cv_count + k] = arr_coefs[ind];
            }
            ++i_model;
        }
    }

    printf("\n");
    printf("n_cv_count %lu\n", n_cv_count);
    // Extract coefficients
    // for (std::int64_t i = 0; i < n_cv_count; ++i) {
    //     printf("%.0lf ", arr_label[i]);
    // }
    // printf("\n");

    return dal::svm::model<Task>()
        .set_biases(dal::detail::homogen_table_builder{}.reset(arr_biases, model_count, 1).build())
        .set_coeffs(dal::detail::homogen_table_builder{}
                        .reset(arr_dual_coef, (class_count - 1), n_cv_count)
                        .build());

    //     auto table_support_vectors =
    //         interop::convert_from_daal_homogen_table<Float>(model.getSupportVectors());
    // auto table_classification_coeffs =
    //     interop::convert_from_daal_homogen_table<Float>(model.getClassificationCoefficients());
    // const double bias = model.getBias();

    // return dal::svm::model<Task>()
    //     .set_support_vectors(table_support_vectors)
    //     .set_coeffs(table_classification_coeffs)
    //     .set_bias(bias);
} // namespace oneapi::dal::svm::backend

} // namespace oneapi::dal::svm::backend
