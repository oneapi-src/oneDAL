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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::triangle_counting {
namespace detail {

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl() {
        if constexpr (std::is_same_v<Task, task::global>) {
            global = true;
        }
        else if constexpr (std::is_same_v<Task, task::local>) {
            local = true;
        }
        else {
            static_assert("Unsupported task");
        }
    }

    bool local = false;
    bool global = false;

    kind _kind;
    relabel _relabel;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
kind descriptor_base<Task>::get_kind() const {
    return impl_->_kind;
}

template <typename Task>
relabel descriptor_base<Task>::get_relabel() const {
    return impl_->_relabel;
}

template <typename Task>
void descriptor_base<Task>::set_kind(kind kind) {
    impl_->_kind = kind;
}

template <typename Task>
void descriptor_base<Task>::set_relabel(relabel relabel) {
    impl_->_relabel = relabel;
}

} // namespace detail
} // namespace oneapi::dal::preview::triangle_counting
