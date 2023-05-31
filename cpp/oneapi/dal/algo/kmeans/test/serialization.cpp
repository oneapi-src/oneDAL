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

#include "oneapi/dal/algo/kmeans/test/data.hpp"
#include "oneapi/dal/algo/kmeans/test/fixture.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/test/engine/metrics/regression.hpp"

namespace oneapi::dal::kmeans::test {

template <typename TestType>
class kmeans_serialization_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = dal::kmeans::method::lloyd_dense;
    using task_t = dal::kmeans::task::clustering;
    using descriptor_t = descriptor<float_t, method_t, task_t>;

    static table get_test_data() {
        constexpr std::int64_t row_count = 5;
        constexpr std::int64_t feature_count = 4;

        static const double x_test[] = {
            -4.7561e+00, -4.5701e+00, 2.0992e-01,  4.9749e-01, //
            -1.8507e+00, 3.5811e+00,  3.5440e+00,  -2.6871e+00, //
            -1.9647e+00, -3.6563e+00, -4.3989e+00, -2.9097e+00, //
            4.8959e+00,  -4.3048e+00, 4.6074e+00,  4.2129e-01, //
            4.4583e+00,  4.2498e+00,  2.7033e+00,  2.4807e+00 //
        };

        return homogen_table::wrap(x_test, row_count, feature_count);
    }

    static table get_train_data() {
        constexpr std::int64_t row_count = 12;
        constexpr std::int64_t feature_count = 4;

        static const double x_train[] = {
            -3.8900e-01, -3.1999e+00, -7.6768e-01, 3.8425e+00, //
            -5.4482e-01, 2.7248e+00,  -1.4848e+00, -3.9388e+00, //
            -3.6133e+00, 3.0425e+00,  1.3176e+00,  4.9769e+00, //
            -2.9177e-01, -5.9228e-01, -3.0740e+00, 3.0995e+00, //
            -4.3426e+00, 4.1916e+00,  3.8152e+00,  -1.3782e+00, //
            1.7607e+00,  -3.5046e+00, 4.5728e+00,  2.5514e+00, //
            2.9286e+00,  -4.8732e+00, -3.2641e+00, -1.5032e+00, //
            -4.5215e+00, -1.0559e+00, -4.6809e+00, 3.4132e+00, //
            -1.9155e+00, -1.2807e+00, -3.0906e+00, 4.4551e+00, //
            -3.8128e+00, -2.6217e+00, -2.4594e-01, -4.1077e+00, //
            4.6350e+00,  -6.9711e-01, -2.6326e-01, -4.0345e+00, //
            -2.8501e+00, 4.9191e+00,  -2.8035e+00, 1.9864e+00 //
        };

        return homogen_table::wrap(x_train, row_count, feature_count);
    }

    static descriptor_t get_descriptor() {
        return descriptor_t();
    }

    model<task_t> train_model() {
        const auto x_train = this->get_train_data();
        return this->train(this->get_descriptor(), x_train).get_model();
    }

    infer_result<task_t> run_inference(const model<task_t>& m) {
        return this->infer(this->get_descriptor(), m, this->get_test_data());
    }

    void compare_infer_results(const infer_result<task_t>& res,
                               const infer_result<task_t>& gtr,
                               double tol = 1e-9) {
        const table& gtr_table = gtr.get_responses();
        const table& res_table = res.get_responses();

        const auto r_count = gtr_table.get_column_count();

        const table scr_table = te::mse_score<float_t>(res_table, gtr_table);

        const auto score = row_accessor<const float_t>(scr_table).pull({ 0, -1 });

        for (std::int64_t r = 0; r < r_count; ++r) {
            REQUIRE(score[r] < tol);
        }
    }

    void run_test() {
        INFO("training");
        const auto model = train_model();

        INFO("serialization");
        const auto deserialized_model = te::serialize_deserialize(model);

        INFO("inference");
        const auto actual = run_inference(model);
        const auto expected = run_inference(deserialized_model);
        compare_infer_results(actual, expected);
    }
};

using kmeans_type = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(kmeans_serialization_test,
                     "serialize/deserialize kmeans models",
                     "[kmeans][test][serialization]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());

    this->run_test();
}

} // namespace oneapi::dal::kmeans::test
