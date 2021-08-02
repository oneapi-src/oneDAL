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

#pragma once

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

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
template <typename Allocator>
class read_args_graph_impl : public base {
public:
    read_args_graph_impl(oneapi::dal::preview::read_mode mode) : mode(mode) {
        if (mode != oneapi::dal::preview::read_mode::edge_list &&
            mode != oneapi::dal::preview::read_mode::weighted_edge_list)
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
    }
    read_args_graph_impl(Allocator alloc, oneapi::dal::preview::read_mode mode)
            : allocator(alloc),
              mode(mode) {
        if (mode != oneapi::dal::preview::read_mode::edge_list &&
            mode != oneapi::dal::preview::read_mode::weighted_edge_list)
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
    }

    Allocator allocator;
    oneapi::dal::preview::read_mode mode;
};
} // namespace detail

struct read_args_tag {};

template <typename Object, typename Allocator>
class ONEDAL_EXPORT read_args : public base {
public:
    using object_t = Object;
    using allocator_t = Allocator;
    using tag_t = read_args_tag;
    read_args(const read_args& args) = default;
    read_args(oneapi::dal::preview::read_mode mode)
            : impl_(new detail::read_args_graph_impl<Allocator>(mode)) {}
    read_args(const Allocator& allocator = std::allocator<char>{},
              oneapi::dal::preview::read_mode mode = oneapi::dal::preview::read_mode::edge_list)
            : impl_(new detail::read_args_graph_impl<Allocator>(mode)) {
        set_allocator_impl(allocator);
    }
    Allocator get_allocator() const {
        return impl_->allocator;
    }

    auto& set_read_mode(oneapi::dal::preview::read_mode mode) {
        set_read_mode_impl(mode);
        return *this;
    }
    oneapi::dal::preview::read_mode get_read_mode() const {
        return impl_->mode;
    }

protected:
    void set_read_mode_impl(oneapi::dal::preview::read_mode mode) {
        if (mode != preview::read_mode::edge_list && mode != preview::read_mode::weighted_edge_list)
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
        impl_->mode = mode;
    }
    void set_allocator_impl(Allocator allocator) {
        impl_->allocator = allocator;
    }

private:
    dal::detail::pimpl<detail::read_args_graph_impl<Allocator>> impl_;
};

// template <typename Object, typename Allocator>
// read_args(Allocator& allocator, oneapi::dal::preview::read_mode mode)->read_args<Object, Allocator>;

} // namespace oneapi::dal::preview::csv
