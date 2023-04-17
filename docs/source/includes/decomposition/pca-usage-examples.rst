.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

Training
--------

::

namespace dal = oneapi::dal;
dal::pca::model<> run_training(const table &data, const std::string &method_name) {
    const auto pca_desc = dal::pca::descriptor<float, Method>{}
                            .set_component_count(5)
                            .set_deterministic(true);
    
    const auto result = dal::train(pca_desc, data);
    
    std::cout << "Method Name:" << method_name << std::endl;
    std::cout << "Eigenvectors:\n"
            << result_train.get_eigenvectors() << std::endl;
    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;
    
    return result.get_model();
}

Inference
---------

::

void run_inference(const pca::model<> &model, const table &new_data) {
    const auto pca_desc = dal::pca::descriptor<float, Method>{}.get_component_count(5)
    
    const auto result = dal::infer(pca_desc, model, new_data);
    
    std::cout << "Transformed data:\n"
            << result_infer.get_transformed_data() << std::endl;
}