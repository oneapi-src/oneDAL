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

#include <include/algorithms/kmeans/kmeans_init_types.h>

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace daal_kmeans_init = daal::algorithms::kmeans::init;

namespace oneapi::dal::kmeans_init::backend {

template <daal_kmeans_init::Method Value>
using daal_method_constant = std::integral_constant<daal_kmeans_init::Method, Value>;

template <typename Method>
struct to_daal_method;

template <>
struct to_daal_method<method::dense> : daal_method_constant<daal_kmeans_init::defaultDense> {};

template <>
struct to_daal_method<method::random_dense> : daal_method_constant<daal_kmeans_init::randomDense> {
};

template <>
struct to_daal_method<method::plus_plus_dense>
        : daal_method_constant<daal_kmeans_init::plusPlusDense> {};

template <>
struct to_daal_method<method::parallel_plus_dense>
        : daal_method_constant<daal_kmeans_init::parallelPlusDense> {};

} // namespace oneapi::dal::kmeans_init::backend
