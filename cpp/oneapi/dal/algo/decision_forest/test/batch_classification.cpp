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

#define sizeofa(p) sizeof(p) / sizeof(*p)

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename TestType>
class df_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = df::task::classification;

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
                           double accuracy_threshold) {
        const auto x_test = data.get_table(data_table_id, range(0, -1));
        const auto y_test =
            data.get_table(data_table_id,
                           range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run inference");
        const auto infer_result = this->infer(desc, model, x_test);
        check_infer_shapes(desc, data, infer_result);

        SECTION("infer accuracy is expected") {
            const auto labels = dal::row_accessor<const Float>(infer_result.get_labels()).pull();
            const auto truth_labels = dal::row_accessor<const Float>(y_test).pull();
            std::int64_t incorrect_label_count = 0;

            for (std::int64_t i = 0; i < labels.get_count(); i++) {
                incorrect_label_count +=
                    (static_cast<int>(labels[i]) != static_cast<int>(truth_labels[i]));
            }
            REQUIRE(static_cast<double>(incorrect_label_count) / labels.get_count() <
                    accuracy_threshold + eps);
        }
        return infer_result;
    }

    void check_train_shapes(const df::descriptor<Float, Method, Task>& desc,
                            const te::dataframe& data,
                            const df::train_result<Task>& result) {
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
        SECTION("infer labels shape is expected") {
            REQUIRE(result.get_labels().get_row_count() == data.get_row_count());
            REQUIRE(result.get_labels().get_column_count() == 1);
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
                    REQUIRE(((required_var_imp_val[i] - var_imp_val[i]) / required_var_imp_val[i]) <
                            accuracy_threshold + eps);
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
                REQUIRE(std::abs((required_oob_error - oob_err_val[0]) / required_oob_error) <
                        accuracy_threshold + eps);
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

                REQUIRE(((oob_err_val[0] - ref_oob_err) / oob_err_val[0]) <
                        accuracy_threshold + eps);
            }
        }
    }

private:
    double eps = 1e-10;
};

struct dataset_info {
    std::string name;
    std::int64_t class_count;
    std::int64_t categ_feature_count = 0;
    const std::int64_t* categ_feature_list = nullptr;
};

struct workload {
    dataset_info ds_info;
    double required_accuracy = 0.0;
};

const std::int64_t df_ds_ion_ftrs_list[] = { 0 };
const dataset_info df_ds_ion = { "ionosphere/dataset/ionosphere",
                                 2 /* class count */,
                                 sizeofa(df_ds_ion_ftrs_list),
                                 df_ds_ion_ftrs_list };
const dataset_info df_ds_segment = { "segment/dataset/segment", 7 /* class count */ };
const dataset_info df_ds_classification = { "classification/dataset/df_classification",
                                            2 /* class count */ };
const dataset_info df_ds_pendigits = { "pendigits/dataset/pendigits", 10 /* class count */ };

using df_types = COMBINE_TYPES((float, double), (df::method::dense, df::method::hist));

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df default long flow",
                     "[df][integration][batch][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());

    const workload wl =
        GENERATE_COPY(workload{ df_ds_ion, 0.95 }, workload{ df_ds_segment, 0.938 });

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 10);
    const bool memory_saving_mode_val = GENERATE_COPY(true, false);

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
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df corner flow",
                     "[df][integration][batch][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = { df_ds_classification };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(10);
    desc.set_error_metric_mode(error_metric_mode::out_of_bag_error);
    desc.set_min_observations_in_leaf_node(8);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df var importance flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->is_gpu()); // var importance differes on GPU due to difference in built model
    SKIP_IF(this->not_available_on_device());
    const workload wl = { df_ds_pendigits };
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

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df small flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = workload{ df_ds_segment, 0.738 };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df impurity flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = workload{ df_ds_segment, 0.738 };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

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
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df all features flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = workload{ df_ds_segment, 0.738 };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

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
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df bootstrap flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = workload{ df_ds_ion, 0.95 };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

    const bool bootstrap_val = GENERATE_COPY(false, true);

    auto desc = this->get_default_descriptor();

    desc.set_bootstrap(bootstrap_val);
    desc.set_max_tree_depth(50);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
}

TEMPLATE_LIST_TEST_M(df_batch_test,
                     "df oob per observation flow",
                     "[df][integration][batch][nightly][external-dataset]",
                     df_types) {
    SKIP_IF(this->not_available_on_device());
    const workload wl = workload{ df_ds_ion, 0.95 };

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe data_test =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".test.csv" });

    const double accuracy_threshold = 1 - wl.required_accuracy;

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
    this->infer_base_checks(desc,
                            data_test,
                            this->get_homogen_table_id(),
                            model,
                            accuracy_threshold);
    this->check_oob_err_matches_oob_err_per_observation(desc, train_result, accuracy_threshold);
}

} // namespace oneapi::dal::decision_forest::test
