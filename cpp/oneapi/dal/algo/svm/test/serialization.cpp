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

#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

namespace oneapi::dal::svm::test {

namespace te = dal::test::engine;

template <typename Float, typename Derived>
class model_serialization_fixture : te::float_algo_fixture<Float> {
public:
};

template <typename TestType>
class svm_serialization_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;
    using kernel_t = dal::linear_kernel::descriptor<float_t>;
    using descriptor_t = descriptor<float_t, method_t, task_t, kernel_t>;

    bool not_available_on_device() {
        constexpr bool is_smo = std::is_same_v<method_t, svm::method::smo>;
        return this->get_policy().is_gpu() && is_smo;
    }

    template <typename T = task_t, detail::enable_if_classification_t<T>* = nullptr>
    void set_class_count(std::int64_t class_count) {
        class_count_ = class_count;
    }

    template <typename T = task_t, detail::enable_if_classification_t<T>* = nullptr>
    std::tuple<table, table> get_train_data() {
        // TODO: Replace by classification dataset generator

        constexpr std::int64_t row_count = 21;
        constexpr std::int64_t feature_count = 3;
        static const float_t x_train[] = {
            -0.543,  0.6576,  0.2046, //
            0.33,    -1.4263, 1.3322, //
            -0.1936, -0.1364, -0.6573, //
            1.1793,  -1.0809, 0.7298, //
            -2.8212, -0.4471, -0.5333, //
            1.122,   0.4834,  0.2969, //
            -2.0703, -1.7256, 0.6822, //
            0.2969,  -1.004,  -0.5835, //
            -1.1765, -2.2248, 0.7409, //
            -1.6578, -2.9339, 1.0975, //
            0.4779,  0.3422,  -1.4021, //
            -0.6799, 0.7451,  -0.7207, //
            -0.086,  -2.1414, -1.8213, //
            -2.3201, -0.8528, -1.3286, //
            -0.2933, 0.5726,  -0.0968, //
            0.0617,  1.0032,  -0.4763, //
            -0.6552, -1.4016, 2.3133, //
            -0.2874, -0.9861, -0.2129, //
            0.1375,  -0.326,  -1.3081, //
            -1.8112, 0.2395,  -0.2247, //
            1.7241,  -0.4099, -1.7277 //
        };

        static const std::int32_t y_train_two_cls[] = { 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1,
                                                        1, 0, 0, 1, 0, 0, 1, 0, 0, 1 };
        static const std::int32_t y_train_three_cls[] = { 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 2,
                                                          1, 0, 2, 2, 2, 0, 1, 0, 0, 1 };

        ONEDAL_ASSERT(class_count_ == 2 || class_count_ == 3);
        const std::int32_t* y_train = (class_count_ == 2) ? y_train_two_cls : y_train_three_cls;

        return { homogen_table::wrap(x_train, row_count, feature_count),
                 homogen_table::wrap(y_train, row_count, 1) };
    }

    template <typename T = task_t, detail::enable_if_regression_t<T>* = nullptr>
    std::tuple<table, table> get_train_data() {
        // TODO: Replace by regression dataset generator

        constexpr std::int64_t row_count = 21;
        constexpr std::int64_t feature_count = 3;
        static const float_t x_train[] = {
            -0.543,  0.6576,  0.2046, //
            0.33,    -1.4263, 1.3322, //
            -0.1936, -0.1364, -0.6573, //
            1.1793,  -1.0809, 0.7298, //
            -2.8212, -0.4471, -0.5333, //
            1.122,   0.4834,  0.2969, //
            -2.0703, -1.7256, 0.6822, //
            0.2969,  -1.004,  -0.5835, //
            -1.1765, -2.2248, 0.7409, //
            -1.6578, -2.9339, 1.0975, //
            0.4779,  0.3422,  -1.4021, //
            -0.6799, 0.7451,  -0.7207, //
            -0.086,  -2.1414, -1.8213, //
            -2.3201, -0.8528, -1.3286, //
            -0.2933, 0.5726,  -0.0968, //
            0.0617,  1.0032,  -0.4763, //
            -0.6552, -1.4016, 2.3133, //
            -0.2874, -0.9861, -0.2129, //
            0.1375,  -0.326,  -1.3081, //
            -1.8112, 0.2395,  -0.2247, //
            1.7241,  -0.4099, -1.7277 //
        };

        static const std::int32_t y_train_two_cls[] = { 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1,
                                                        1, 0, 0, 1, 0, 0, 1, 0, 0, 1 };
        static const std::int32_t y_train_three_cls[] = { 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 2,
                                                          1, 0, 2, 2, 2, 0, 1, 0, 0, 1 };

        ONEDAL_ASSERT(class_count_ == 2 || class_count_ == 3);
        const std::int32_t* y_train = (class_count_ == 2) ? y_train_two_cls : y_train_three_cls;

        return { homogen_table::wrap(x_train, row_count, feature_count),
                 homogen_table::wrap(y_train, row_count, 1) };
    }

    template <typename T = task_t, detail::enable_if_classification_t<T>* = nullptr>
    table get_test_data() {
        constexpr std::int64_t row_count = 9;
        constexpr std::int64_t feature_count = 3;
        static const float_t x_test[] = {
            0.4213,  1.5162,  0.1232, //
            -0.9252, -0.514,  -0.4708, //
            0.87,    0.8338,  3.4986, //
            1.1788,  -0.5077, 4.5583, //
            -1.4726, 2.2684,  -1.8172, //
            -0.0091, -1.0776, 1.5261, //
            -0.5016, -2.9235, 0.3886, //
            0.4889,  -0.2868, 0.6805, //
            -2.0861, 0.9992,  -1.9605, //
        };
        return homogen_table::wrap(x_test, row_count, feature_count);
    }

    template <typename T = task_t, detail::enable_if_classification_t<T>* = nullptr>
    descriptor_t get_descriptor() {
        ONEDAL_ASSERT(class_count_ > 0);
        return descriptor_t{}.set_class_count(class_count_);
    }

    template <typename T = task_t, detail::enable_if_regression_t<T>* = nullptr>
    descriptor_t get_descriptor() {
        return descriptor_t{};
    }

    model<task_t> train_model() {
        const auto [x_train, y_train] = this->get_train_data();
        return this->train(this->get_descriptor(), x_train, y_train).get_model();
    }

    infer_result<task_t> run_inference(const model<task_t>& m) {
        return this->infer(this->get_descriptor(), m, this->get_test_data());
    }

    void compare_infer_results(const infer_result<task_t>& actual,
                               const infer_result<task_t>& reference) {
        INFO("compare labels") {
            te::check_if_tables_equal<float_t>(actual.get_labels(), reference.get_labels());
        }
        INFO("compare decision function") {
            te::check_if_tables_equal<float_t>(actual.get_decision_function(),
                                               reference.get_decision_function());
        }
    }

    void run_test() {
        INFO("training");
        const auto model = train_model();

        INFO("serialization");
        const auto deserialized_model = te::serialize_deserialize(model);

        INFO("inference");
        const auto expected = run_inference(model);
        const auto actual = run_inference(deserialized_model);
        compare_infer_results(actual, expected);
    }

private:
    std::int64_t class_count_ = -1;
};

using svm_cls_types = COMBINE_TYPES((float, double),
                                    (svm::method::thunder, svm::method::smo),
                                    (svm::task::classification));

using svm_reg_types = COMBINE_TYPES((float, double),
                                    (svm::method::thunder, svm::method::smo),
                                    (svm::task::regression));

TEMPLATE_LIST_TEST_M(svm_serialization_test,
                     "serialize/deserialize classification svm model",
                     "[cls]",
                     svm_cls_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    const std::int64_t class_count = GENERATE(2, 3);

    // Multiclass is not supported on GPU
    SKIP_IF(this->get_policy().is_gpu() && class_count > 2);

    this->set_class_count(class_count);
    this->run_test();
}

TEMPLATE_LIST_TEST_M(svm_serialization_test,
                     "serialize/deserialize regression svm model",
                     "[reg]",
                     svm_reg_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    this->run_test();
}

} // namespace oneapi::dal::svm::test
