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

#include "oneapi/dal/algo/decision_forest/train.hpp"
#include "oneapi/dal/algo/decision_forest/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include <list>

#define sizeofa(p) sizeof(p) / sizeof(*p)

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename T>
struct checker_info {
    typedef T (*checker_func)(const dal::v1::table& infer_labels,
                              const dal::v1::table& ground_truth);

    std::string name;
    checker_func check;
    double required_accuracy;
};

template <typename TestType>
class df_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = std::tuple_element_t<2, TestType>;

    bool is_gpu() {
        return get_policy().is_gpu();
    }

    bool not_available_on_device() {
        constexpr bool is_dense = std::is_same_v<Method, decision_forest::method::dense>;
        return get_policy().is_gpu() && is_dense;
    }

    auto get_default_descriptor() {
        return df::descriptor<Float, Method, Task>{};
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    auto get_cls_dataframe_base() {
        constexpr double required_accuracy = 0.95;
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t row_count_test = 3;
        constexpr std::int64_t column_count = 3;
        constexpr std::int64_t class_count = 2;

        static const float train_arr[] = { -2.f, -1.f, 0.f, -1.f, -1.f, 0.f, -1.f, -2.f, 0.f,
                                           +1.f, +1.f, 1.f, +1.f, +2.f, 1.f, +2.f, +1.f, 1.f };

        static const float test_arr[] = { -1.f, -1.f, 0.f, +2.f, +2.f, 1.f, +3.f, +2.f, 1.f };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count),
                            row_count_train,
                            column_count };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                 row_count_test,
                                 column_count };

        const std::list<checker_info<double>> checker_list = { this->get_cls_checker(
            1 - required_accuracy) };

        return std::make_tuple(data, data_test, class_count, checker_list);
    }

    auto get_cls_dataframe(std::string ds_name, double required_accuracy) {
        const te::dataframe data =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".train.csv" });
        const te::dataframe data_test =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".test.csv" });

        const std::list<checker_info<double>> checker_list = { this->get_cls_checker(
            1 - required_accuracy) };
        return std::make_tuple(data, data_test, checker_list);
    }

    auto get_reg_dataframe_base() {
        const double required_mse = 0.05;
        constexpr std::int64_t row_count_train = 10;
        constexpr std::int64_t row_count_test = 5;
        constexpr std::int64_t column_count = 3;

        static const float train_arr[] = {
            0.1f,    0.25f,   0.0079f, 0.15f,   0.35f,   0.0160f, 0.25f,   0.55f,
            0.0407f, 0.3f,    0.65f,   0.0573f, 0.4f,    0.85f,   0.0989f, 0.45f,
            0.95f,   0.1240f, 0.55f,   1.15f,   0.1827f, 0.6f,    1.25f,   0.2163f,
            0.7f,    1.45f,   0.2919f, 0.8f,    1.65f,   0.3789f,
        };

        static const float test_arr[] = {
            0.2f,    0.45f, 0.0269f, 0.35f,   0.75f, 0.0767f, 0.5f,    1.05f,
            0.1519f, 0.65f, 1.35f,   0.2527f, 0.75f, 1.55f,   0.3340f,
        };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count),
                            row_count_train,
                            column_count };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                 row_count_test,
                                 column_count };

        const std::list<checker_info<double>> checker_list = { this->get_mse_checker(
            required_mse) };

        return std::make_tuple(data, data_test, checker_list);
    }

    auto get_reg_dataframe(std::string ds_name, double required_mse, double required_mae) {
        const te::dataframe data =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".train.csv" });
        const te::dataframe data_test =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".test.csv" });

        const std::list<checker_info<double>> checker_list = { this->get_mse_checker(required_mse),
                                                               this->get_mae_checker(
                                                                   required_mae) };

        return std::make_tuple(data, data_test, checker_list);
    }

    auto train_base_checks(const df::descriptor<Float, Method, Task>& desc,
                           const te::dataframe& data,
                           const te::table_id& data_table_id) {
        const auto x = data.get_table(data_table_id, range(0, -1));
        const auto y = data.get_table(data_table_id,
                                      range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run training");
        const auto train_result = this->train(desc, x, y);
        check_train_shapes(desc, data, train_result);
        return train_result;
    }

    auto infer_base_checks(const df::descriptor<Float, Method, Task>& desc,
                           const te::dataframe& data,
                           const te::table_id& data_table_id,
                           const df::model<Task>& model,
                           const std::list<checker_info<double>>& checker_list) {
        const auto x_test = data.get_table(data_table_id, range(0, -1));
        const auto y_test =
            data.get_table(data_table_id,
                           range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run inference");
        const auto infer_result = this->infer(desc, model, x_test);
        check_infer_shapes(desc, data, infer_result);

        SECTION("infer accuracy is expected") {
            for (auto ch : checker_list) {
                CAPTURE(ch.name);
                REQUIRE(ch.check(infer_result.get_labels(), y_test) < ch.required_accuracy + eps);
            }
        }
        return infer_result;
    }

    void check_train_shapes(const df::descriptor<Float, Method, Task>& desc,
                            const te::dataframe& data,
                            const df::train_result<Task>& result) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;

        SECTION("model shape is expected") {
            REQUIRE(result.get_model().get_tree_count() == desc.get_tree_count());
            if constexpr (is_cls) {
                REQUIRE(result.get_model().get_class_count() == desc.get_class_count());
            }
        }

        if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
            SECTION("oob error shape is expected") {
                REQUIRE(result.get_oob_err().has_data());
                REQUIRE(result.get_oob_err().get_row_count() == 1);
                REQUIRE(result.get_oob_err().get_column_count() == 1);
            }
        }

        if (check_mask_flag(desc.get_error_metric_mode(),
                            error_metric_mode::out_of_bag_error_per_observation)) {
            SECTION("oob error per observation shape is expected") {
                REQUIRE(result.get_oob_err_per_observation().has_data());
                REQUIRE(result.get_oob_err_per_observation().get_row_count() ==
                        data.get_row_count());
                REQUIRE(result.get_oob_err_per_observation().get_column_count() == 1);
            }
        }

        if (variable_importance_mode::none != desc.get_variable_importance_mode()) {
            SECTION("variable improtance shape is expected") {
                REQUIRE(result.get_var_importance().has_data());
                REQUIRE(result.get_var_importance().get_row_count() == 1);
                REQUIRE(result.get_var_importance().get_column_count() ==
                        data.get_column_count() - 1);
            }
        }
    }

    void check_infer_shapes(const df::descriptor<Float, Method, Task>& desc,
                            const te::dataframe& data,
                            const df::infer_result<Task>& result) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;
        if constexpr (is_cls) {
            if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_labels)) {
                SECTION("infer labels shape is expected") {
                    REQUIRE(result.get_labels().has_data());
                    REQUIRE(result.get_labels().get_row_count() == data.get_row_count());
                    REQUIRE(result.get_labels().get_column_count() == 1);
                }
            }

            if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
                SECTION("infer probabilities shape is expected") {
                    REQUIRE(result.get_probabilities().has_data());
                    REQUIRE(result.get_probabilities().get_row_count() == data.get_row_count());
                    REQUIRE(result.get_probabilities().get_column_count() ==
                            desc.get_class_count());
                }
            }
        }
        else {
            SECTION("infer labels shape is expected") {
                REQUIRE(result.get_labels().has_data());
                REQUIRE(result.get_labels().get_row_count() == data.get_row_count());
                REQUIRE(result.get_labels().get_column_count() == 1);
            }
        }
    }

    void check_var_importance_matches_required(const df::descriptor<Float, Method, Task>& desc,
                                               const df::train_result<Task>& train_result,
                                               const te::dataframe& var_imp_data,
                                               const te::table_id& data_table_id,
                                               double accuracy_threshold) {
        if (variable_importance_mode::none != desc.get_variable_importance_mode()) {
            SECTION("match of variable importance vs required one is expected") {
                const auto required_var_imp = var_imp_data.get_table(data_table_id);
                std::int64_t row_ind = 0;
                switch (desc.get_variable_importance_mode()) {
                    case variable_importance_mode::mda_raw: row_ind = 1; break;
                    case variable_importance_mode::mda_scaled: row_ind = 2; break;
                    default: row_ind = 0; break;
                };

                const auto var_imp_val =
                    dal::row_accessor<const Float>(train_result.get_var_importance()).pull();
                const auto required_var_imp_val =
                    dal::row_accessor<const float>(required_var_imp).pull({ row_ind, row_ind + 1 });

                for (std::int32_t i = 0; i < var_imp_val.get_count(); i++) {
                    if (required_var_imp_val[i] > 0.0) {
                        REQUIRE(((required_var_imp_val[i] - var_imp_val[i]) /
                                 required_var_imp_val[i]) < accuracy_threshold + eps);
                    }
                }
            }
        }
    }

    void check_oob_err_matches_required(const df::descriptor<Float, Method, Task>& desc,
                                        const df::train_result<Task>& train_result,
                                        double required_oob_error,
                                        double accuracy_threshold) {
        if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
            SECTION("match of oob error vs required one is expected") {
                const auto oob_err_val =
                    dal::row_accessor<const double>(train_result.get_oob_err()).pull();
                if (required_oob_error > 0.0) {
                    REQUIRE(std::abs((required_oob_error - oob_err_val[0]) / required_oob_error) <
                            accuracy_threshold + eps);
                }
            }
        }
    }

    void check_oob_err_matches_oob_err_per_observation(
        const df::descriptor<Float, Method, Task>& desc,
        const df::train_result<Task>& train_result,
        double accuracy_threshold) {
        if (check_mask_flag(desc.get_error_metric_mode(),
                            error_metric_mode::out_of_bag_error_per_observation)) {
            SECTION("match of oob error vs cumulative oob error per observation is expected") {
                const auto oob_err_val =
                    dal::row_accessor<const double>(train_result.get_oob_err()).pull();
                const auto oob_err_per_obs_arr =
                    dal::row_accessor<const double>(train_result.get_oob_err_per_observation())
                        .pull();

                std::int64_t oob_err_obs_count = 0;
                double ref_oob_err = 0.0;
                for (std::int64_t i = 0; i < oob_err_per_obs_arr.get_count(); i++) {
                    if (oob_err_per_obs_arr[i] >= 0.0) {
                        oob_err_obs_count++;
                        ref_oob_err += oob_err_per_obs_arr[i];
                    }
                    else {
                        REQUIRE(oob_err_per_obs_arr[i] >= -1.0);
                    }
                }

                if (oob_err_val[0] > 0.0) {
                    REQUIRE(((oob_err_val[0] - ref_oob_err) / oob_err_val[0]) <
                            accuracy_threshold + eps);
                }
            }
        }
    }

    checker_info<double> get_cls_checker(double required_accuracy) {
        return checker_info<double>{ "cls_checker",
                                     &calculate_classification_error,
                                     required_accuracy };
    }

    checker_info<double> get_mse_checker(double required_accuracy) {
        return checker_info<double>{ "mse_checker", &calculate_mse, required_accuracy };
    }

    checker_info<double> get_mae_checker(double required_accuracy) {
        return checker_info<double>{ "mae_checker", &calculate_mae, required_accuracy };
    }

    static double calculate_classification_error(const dal::table& infer_labels,
                                                 const dal::table& ground_truth) {
        const auto labels = dal::row_accessor<const Float>(infer_labels).pull();
        const auto truth_labels = dal::row_accessor<const Float>(ground_truth).pull();
        std::int64_t incorrect_label_count = 0;

        for (std::int64_t i = 0; i < labels.get_count(); i++) {
            incorrect_label_count +=
                (static_cast<int>(labels[i]) != static_cast<int>(truth_labels[i]));
        }
        return static_cast<double>(incorrect_label_count) / labels.get_count();
    }

    static double calculate_mse(const dal::v1::table& infer_labels,
                                const dal::v1::table& ground_truth) {
        double mean = 0.0;
        const auto labels = dal::row_accessor<const Float>(infer_labels).pull();
        const auto truth_labels = dal::row_accessor<const Float>(ground_truth).pull();
        for (std::int64_t i = 0; i < labels.get_count(); i++) {
            mean += (labels[i] - truth_labels[i]) * (labels[i] - truth_labels[i]);
        }

        return mean / labels.get_count();
    }

    static double calculate_mae(const dal::v1::table& infer_labels,
                                const dal::v1::table& ground_truth) {
        double mae = 0.0;
        const auto labels = dal::row_accessor<const Float>(infer_labels).pull();
        const auto truth_labels = dal::row_accessor<const Float>(ground_truth).pull();

        for (std::int64_t i = 0; i < labels.get_count(); i++) {
            mae += std::abs(labels[i] - truth_labels[i]);
        }

        return mae / labels.get_count();
    }

private:
    double eps = 1e-10;
};

struct dataset_info {
    std::string name;
    std::int64_t class_count = 0;
    std::int64_t categ_feature_count = 0;
    const std::int64_t* categ_feature_list = nullptr;
};

struct workload_cls {
    dataset_info ds_info;
    double required_accuracy = 0.0;
};

struct workload_reg {
    dataset_info ds_info;
    double required_mse = 0.0;
    double required_mae = 0.0;
};

// dataset configuration
const std::int64_t df_ds_ion_ftrs_list[] = { 0 };
const dataset_info df_ds_ion = { "ionosphere/dataset/ionosphere",
                                 2 /* class count */,
                                 sizeofa(df_ds_ion_ftrs_list),
                                 df_ds_ion_ftrs_list };
const dataset_info df_ds_segment = { "segment/dataset/segment", 7 /* class count */ };
const dataset_info df_ds_classification = { "classification/dataset/df_classification",
                                            2 /* class count */ };
const dataset_info df_ds_pendigits = { "pendigits/dataset/pendigits", 10 /* class count */ };

const dataset_info df_ds_white_wine = { "white_wine/dataset/white_wine" };

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));
using df_reg_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::regression));

#define DF_BATCH_CLS_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_batch_test, name, "[df][integration][batch]", df_cls_types)
#define DF_BATCH_CLS_TEST_EXT(name)                                             \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                         \
                         name,                                                  \
                         "[df][integration][batch][nightly][external-dataset]", \
                         df_cls_types)

#define DF_BATCH_REG_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_batch_test, name, "[df][integration][batch]", df_reg_types)
#define DF_BATCH_REG_TEST_EXT(name)                                             \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                         \
                         name,                                                  \
                         "[df][integration][batch][nightly][external-dataset]", \
                         df_reg_types)

DF_BATCH_CLS_TEST_EXT("df cls default flow") {
    SKIP_IF(this->not_available_on_device());

    const workload_cls wl =
        GENERATE_COPY(workload_cls{ df_ds_ion, 0.95 }, workload_cls{ df_ds_segment, 0.938 });

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 10);
    const bool memory_saving_mode_val = this->is_gpu() ? false : GENERATE_COPY(true, false);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;

    auto desc = this->get_default_descriptor();

    desc.set_memory_saving_mode(memory_saving_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_features_per_node(features_per_node_val);
    desc.set_max_tree_depth(max_tree_depth_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls corner flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_classification, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(10);
    desc.set_error_metric_mode(error_metric_mode::out_of_bag_error);
    desc.set_min_observations_in_leaf_node(8);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df var importance flow") {
    SKIP_IF(this->is_gpu()); // var importance differes on GPU due to difference in built model
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_pendigits };
    const double oob_required_accuracy = 0.65;
    const double oob_required_error = 0.00867361;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe var_imp_test_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.var_imp.csv" });

    const double accuracy_threshold = 1 - oob_required_accuracy;

    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val = GENERATE_COPY(variable_importance_mode::none,
                                                            variable_importance_mode::mdi,
                                                            variable_importance_mode::mda_raw,
                                                            variable_importance_mode::mda_scaled);

    auto desc = this->get_default_descriptor();

    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());

    this->check_oob_err_matches_required(desc,
                                         train_result,
                                         oob_required_error,
                                         accuracy_threshold);
    this->check_oob_err_matches_oob_err_per_observation(desc, train_result, accuracy_threshold);
    this->check_var_importance_matches_required(desc,
                                                train_result,
                                                var_imp_test_data,
                                                this->get_homogen_table_id(),
                                                accuracy_threshold);
}

DF_BATCH_CLS_TEST_EXT("df cls small flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls impurity flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;
    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(500);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_min_observations_in_leaf_node(30);
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls all features flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(30);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_features_per_node(data.get_column_count() - 1); // skip labels column
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls bootstrap flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const bool bootstrap_val = GENERATE_COPY(false, true);

    auto desc = this->get_default_descriptor();

    desc.set_bootstrap(bootstrap_val);
    desc.set_max_tree_depth(50);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls oob per observation flow") {
    SKIP_IF(this->not_available_on_device());
    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val = GENERATE_COPY(variable_importance_mode::none,
                                                            variable_importance_mode::mdi,
                                                            variable_importance_mode::mda_raw,
                                                            variable_importance_mode::mda_scaled);
    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const double observations_per_tree_fraction_val = GENERATE_COPY(1.0, 0.5);

    auto desc = this->get_default_descriptor();

    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_features_per_node(features_per_node_val);
    desc.set_max_tree_depth(10);
    desc.set_observations_per_tree_fraction(observations_per_tree_fraction_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_oob_err_matches_oob_err_per_observation(desc,
                                                        train_result,
                                                        1 - wl.required_accuracy);
}

DF_BATCH_CLS_TEST("df cls base check with default paarams") {
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    auto desc = this->get_default_descriptor();

    desc.set_class_count(class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST("df cls base check with non default paarams") {
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);
    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val =
        GENERATE_COPY(variable_importance_mode::none, variable_importance_mode::mdi);
    const auto infer_mode_val =
        GENERATE_COPY(df::infer_mode::class_labels,
                      df::infer_mode::class_labels | df::infer_mode::class_probabilities);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);
    desc.set_min_observations_in_leaf_node(2);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_infer_mode(infer_mode_val);
    desc.set_voting_mode(df::voting_mode::unweighted);
    desc.set_class_count(class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// regression tests

DF_BATCH_REG_TEST("df reg base check with default paarams") {
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    auto desc = this->get_default_descriptor();

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST("df reg base check with non default paarams") {
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);
    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val =
        GENERATE_COPY(variable_importance_mode::none, variable_importance_mode::mdi);
    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);
    desc.set_min_observations_in_leaf_node(2);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_EXT("df reg default flow") {
    SKIP_IF(this->not_available_on_device());

    const workload_reg wl = { df_ds_white_wine, 0.45, 0.5 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    auto desc = this->get_default_descriptor();

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_EXT("df reg small flow") {
    SKIP_IF(this->not_available_on_device());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(tree_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_EXT("df reg impurity flow") {
    SKIP_IF(this->not_available_on_device());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(500);
    desc.set_min_observations_in_leaf_node(30);
    desc.set_impurity_threshold(impurity_threshold_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_EXT("df reg bootstrap  flow") {
    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 50);
    const bool bootstrap_val = GENERATE_COPY(false, true);

    auto desc = this->get_default_descriptor();
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_max_tree_depth(max_tree_depth_val);
    desc.set_bootstrap(bootstrap_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

} // namespace oneapi::dal::decision_forest::test
