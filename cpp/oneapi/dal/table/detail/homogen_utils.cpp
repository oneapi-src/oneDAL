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

#include "oneapi/dal/table/detail/homogen_utils.hpp"
#include "oneapi/dal/table/backend/homogen_table_impl.hpp"

namespace oneapi::dal::detail::v1 {

array<byte_t> get_original_data(const homogen_table& t) {
    using impl_t = backend::homogen_table_impl;
    using wrapper_t = homogen_table_impl_wrapper<impl_t>;

    auto* impl_raw_ptr = dynamic_cast<wrapper_t*>(&get_impl(t));
    if (impl_raw_ptr != nullptr) {
        auto& homogen_impl = impl_raw_ptr->get();
        return homogen_impl.get_data_array();
    }
    else {
        return array<byte_t>{};
    }
}

} // namespace oneapi::dal::detail::v1
