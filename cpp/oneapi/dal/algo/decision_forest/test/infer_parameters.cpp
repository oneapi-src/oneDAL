/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::decision_forest::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class df_infer_params_test : public df_test<TestType, df_infer_params_test<TestType>> {
public:
    using base_t = df_test<TestType, df_infer_params_test<TestType>>;

    using task_t = typename base_t::task_t;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;
    using descriptor_t = typename base_t::descriptor_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;
    using infer_input_t = typename base_t::infer_input_t;
    using infer_result_t = typename base_t::infer_result_t;
    using model_t = typename base_t::model_t;

    void generate_parameters() {
        this->block_ = GENERATE(32, 64);
        this->min_trees_for_threading_ = GENERATE(50, 120);
        this->min_number_of_rows_for_vect_seq_compute_ = GENERATE(32);
        this->scale_factor_for_vect_par_compute_ = GENERATE(0.3);
        this->pack_as_struct_ = GENERATE(0, 1);
    }

    auto get_current_parameters() const {
        detail::infer_parameters res{};
        res.set_block_size(this->block_);
        res.set_min_trees_for_threading(this->min_trees_for_threading_);
        res.set_min_number_of_rows_for_vect_seq_compute(this->min_number_of_rows_for_vect_seq_compute_);
        res.set_scale_factor_for_vect_parallel_compute(this->scale_factor_for_vect_par_compute_);

        return res;
    }

    template <typename Desc, typename... Args>
    infer_result_t infer_override(Desc&& desc, Args&&... args) {
        REQUIRE(this->block_ > 0);
        REQUIRE(this->min_trees_for_threading_ > 0);
        REQUIRE(this->min_number_of_rows_for_vect_seq_compute_ > 0);
        REQUIRE(0 < this->scale_factor_for_vect_par_compute_);
        REQUIRE(this->scale_factor_for_vect_par_compute_ < 1.0);
        const auto params = this->get_current_parameters();

        if (this->pack_as_struct_) {
            return te::float_algo_fixture<float_t>::infer(
                std::forward<Desc>(desc),
                params,
                infer_input_t{ std::forward<Args>(args)... });
        }
        else {
            return te::float_algo_fixture<float_t>::infer(std::forward<Desc>(desc),
                                                          params,
                                                          std::forward<Args>(args)...);
        }
    }

private:
    std::int64_t block_;
    std::int64_t min_trees_for_threading_;
    std::int64_t min_number_of_rows_for_vect_seq_compute_;
    double scale_factor_for_vect_par_compute_;
    bool pack_as_struct_;
};

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));

// future inprovements: remove code duplication in this test and in batch.cpp

// dataset configuration
const std::int64_t df_ds_ion_ftrs_list[] = { 0 };
const dataset_info df_ds_ion = { "workloads/ionosphere/dataset/ionosphere",
                                 2 /* class count */,
                                 sizeofa(df_ds_ion_ftrs_list),
                                 df_ds_ion_ftrs_list };
const dataset_info df_ds_segment = { "workloads/segment/dataset/segment", 7 /* class count */ };

TEMPLATE_LIST_TEST_M(df_infer_params_test, "DF classification infer params", "[df][cls][infer][params]", df_cls_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

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

    this->generate_parameters();

    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

} // namespace oneapi::dal::decision_forest::test
