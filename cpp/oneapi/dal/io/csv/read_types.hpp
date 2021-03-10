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

template <typename Allocator>
class read_args_graph_impl : public base {
public:
    read_args_graph_impl(preview::read_mode mode = preview::read_mode::edge_list) : mode(mode) {
        if (mode != preview::read_mode::edge_list)
            throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
    }

    preview::read_mode mode;
    Allocator allocator;
};
} // namespace v1

using v1::read_args_impl;
using v1::read_args_graph_impl;

} // namespace detail

namespace v1 {

template <typename Graph = table, typename Allocator = std::allocator<char>>
class ONEDAL_EXPORT read_args : public base {
public:
    read_args() : impl_(new detail::read_args_graph_impl<Allocator>()) {}
    read_args(preview::read_mode mode) : impl_(new detail::read_args_graph_impl<Allocator>(mode)) {}

    auto& set_allocator(Allocator allocator) {
        set_allocator_impl(allocator);
        return *this;
    }
    Allocator get_allocator() const;

    auto& set_read_mode(oneapi::dal::preview::read_mode mode) {
        set_read_mode_impl(mode);
        return *this;
    }
    oneapi::dal::preview::read_mode get_read_mode() const;

protected:
    void set_read_mode_impl(oneapi::dal::preview::read_mode mode);
    void set_allocator_impl(Allocator allocator);

private:
    dal::detail::pimpl<detail::read_args_graph_impl<Allocator>> impl_;
};

template <>
class ONEDAL_EXPORT read_args<table, std::allocator<char>> : public base {
public:
    read_args();
    read_args(oneapi::dal::preview::read_mode mode);
    auto& set_read_mode(oneapi::dal::preview::read_mode mode) {
        set_read_mode_impl(mode);
        return *this;
    }
    oneapi::dal::preview::read_mode get_read_mode();

protected:
    void set_read_mode_impl(oneapi::dal::preview::read_mode mode);

private:
    dal::detail::pimpl<detail::read_args_impl<table>> impl_;
};

template <typename Graph, typename Allocator>
preview::read_mode read_args<Graph, Allocator>::get_read_mode() const {
    return impl_->mode;
}

template <typename Graph, typename Allocator>
void read_args<Graph, Allocator>::set_read_mode_impl(preview::read_mode mode) {
    if (mode != preview::read_mode::edge_list)
        throw invalid_argument(dal::detail::error_messages::unsupported_read_mode());
    impl_->mode = mode;
}

template <typename Graph, typename Allocator>
Allocator read_args<Graph, Allocator>::get_allocator() const {
    return impl_->allocator;
}

template <typename Graph, typename Allocator>
void read_args<Graph, Allocator>::set_allocator_impl(Allocator allocator) {
    impl_->allocator = allocator;
}

} // namespace v1

using v1::read_args;

} // namespace oneapi::dal::csv
