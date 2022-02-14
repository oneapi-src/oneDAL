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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::csv {

namespace detail {
namespace v1 {
template <typename Object>
class read_args_impl;
} // namespace v1

using v1::read_args_impl;

} // namespace detail

namespace v1 {

template <typename Object = table>
class read_args;

template <>
class ONEDAL_EXPORT read_args<table> : public base {
public:
    read_args();

private:
    dal::detail::pimpl<detail::read_args_impl<table>> impl_;
};

} // namespace v1

using v1::read_args;

} // namespace oneapi::dal::csv

namespace oneapi::dal::preview::csv {

namespace detail {

using byte_alloc_ptr = dal::detail::shared<preview::detail::byte_alloc_iface>;

template <typename Allocator>
inline byte_alloc_ptr make_allocator(Allocator&& alloc) {
    return std::make_shared<preview::detail::alloc_connector<std::decay_t<Allocator>>>(
        std::forward<Allocator>(alloc));
}

class read_args_graph_impl : public base {
public:
    explicit read_args_graph_impl(const byte_alloc_ptr& alloc, read_mode mode)
            : allocator(alloc),
              mode(mode) {
        if (mode != read_mode::edge_list && mode != read_mode::weighted_edge_list) {
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
        }
    }

    byte_alloc_ptr get_allocator() const {
        return allocator;
    }
    read_mode get_read_mode() const {
        return mode;
    }
    byte_alloc_ptr allocator;
    read_mode mode;
};
} // namespace detail

struct read_args_tag {};

template <typename Object>
class ONEDAL_EXPORT read_args : public base {
public:
    using object_t = Object;
    using tag_t = read_args_tag;
    explicit read_args(read_mode mode = read_mode::edge_list)
            : impl_(new detail::read_args_graph_impl(detail::make_allocator(std::allocator<int>{}),
                                                     mode)) {}
    template <typename Allocator>
    explicit read_args(Allocator&& allocator, read_mode mode = read_mode::edge_list)
            : impl_(new detail::read_args_graph_impl(
                  detail::make_allocator(std::forward<Allocator>(allocator)),
                  mode)) {}

    detail::byte_alloc_ptr get_allocator() const {
        return impl_->get_allocator();
    }

    auto& set_read_mode(read_mode mode) {
        set_read_mode_impl(mode);
        return *this;
    }
    read_mode get_read_mode() const {
        return impl_->get_read_mode();
    }

protected:
    void set_read_mode_impl(read_mode mode) {
        if (mode != read_mode::edge_list && mode != read_mode::weighted_edge_list)
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
        impl_->mode = mode;
    }

private:
    dal::detail::pimpl<detail::read_args_graph_impl> impl_;
};

} // namespace oneapi::dal::preview::csv
