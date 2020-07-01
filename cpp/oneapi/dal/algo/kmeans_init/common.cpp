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

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init {

class detail::descriptor_impl : public base {
public:
    std::int64_t cluster_count = -1;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

std::int64_t descriptor_base::get_cluster_count() const {
    return impl_->cluster_count;
}

void descriptor_base::set_cluster_count_impl(std::int64_t value) {
    impl_->cluster_count = value;
}

} // namespace oneapi::dal::kmeans_init
