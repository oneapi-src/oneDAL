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
#include <iostream>
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

template <typename Method>
void run(
    const dal::table& x_train,
    const dal::table& ev,
    const dal::table& mns,
    const dal::table& vrs, 
    const std::string& method_name) {
    const auto pca_desc =
        dal::pca::descriptor<float, Method>().set_component_count(2).set_deterministic(true);

    const auto result_train = dal::train(pca_desc, x_train);

    std::cout << method_name << "\n" << std::endl;

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;


    const auto result_infer = dal::infer(pca_desc, result_train.get_model(), x_train);

    std::cout << "Before Transformed data:\n" << result_infer.get_transformed_data() << std::endl;

    auto model = result_train.get_model();

    // model.set_eigenvalues(ev);

    model.set_means(mns);

    model.set_variances(vrs);

    std::cout << "Eigenvectors:\n" << model.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << model.get_eigenvalues() << std::endl;

    std::cout << "Means:\n" << model.get_means() << std::endl;

    std::cout << "Variances:\n" << model.get_variances() << std::endl;

    const auto result_infer2 = dal::infer(pca_desc, model, x_train);

    std::cout << "After Transformed data:\n" << result_infer2.get_transformed_data() << std::endl;
}

int main(int argc, char const* argv[]) {
    const auto train_name = get_data_path("c_data.csv");
    const auto ev_name = get_data_path("c_ev.csv");
    const auto mn_name = get_data_path("c_mns.csv");
    const auto vr_name = get_data_path("c_vrs.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_name });

    const auto ev = dal::read<dal::table>(dal::csv::data_source{ ev_name });

    const auto mns = dal::read<dal::table>(dal::csv::data_source{ mn_name });

    const auto vrs = dal::read<dal::table>(dal::csv::data_source{ vr_name });

    run<dal::pca::method::cov>(x_train, ev, mns, vrs, "Training method: Covariance");
    run<dal::pca::method::svd>(x_train, ev, mns, vrs, "Training method: SVD");

    return 0;
}
