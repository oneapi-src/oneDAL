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

#pragma once

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend::primitives {

/// Class that hides the implementation details of the `sparse_matrix_handle` class
class sparse_matrix_handle_iface;

/// Handle that is used to store the information about the data in starse format
class sparse_matrix_handle {
    friend detail::pimpl_accessor;

public:
    sparse_matrix_handle();

private:
    detail::pimpl<sparse_matrix_handle_iface> impl_;
};
}
