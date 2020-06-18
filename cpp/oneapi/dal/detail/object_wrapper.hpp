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

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal {
namespace detail {

class object_wrapper_iface {
  public:
    ~object_wrapper_iface() = default;
};

template <typename Object>
class object_wrapper_impl : public base,
                            public object_wrapper_iface {
  public:
    explicit object_wrapper_impl(const Object& object)
        : object_(object) {}

    const Object& get_wrapped() const {
        return object_;
    }

    Object& get_wrapped() {
        return object_;
    }

  private:
    Object object_;
};

class object_wrapper {
  public:
    object_wrapper() = default;

    template <typename Object>
    explicit object_wrapper(Object&& object)
        : impl_(new object_wrapper_impl<Object>{object}) {}

    template <typename Object>
    const Object& get_wrapped() const {
        const auto wrapper = static_cast<const object_wrapper_impl<Object>&>(*impl_);
        return wrapper.get_wrapped();
    }

    template <typename Object>
    Object& get_wrapped() {
        auto wrapper = static_cast<object_wrapper_impl<Object>&>(*impl_);
        return wrapper.get_wrapped();
    }

  private:
    pimpl<object_wrapper_iface> impl_;
};

template <typename Object>
object_wrapper wrap_object(Object&& object) {
    return object_wrapper(object);
}

template <typename Object>
const Object& unwrap_object(const object_wrapper& wrapper) {
    return wrapper.get_wrapped<Object>();
}

} // namespace detail
} // namespace oneapi::dal
