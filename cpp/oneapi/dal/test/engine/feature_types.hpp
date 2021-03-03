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

#pragma once

#include <map>
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

class feature_types_impl {
public:
    explicit feature_types_impl(const array<feature_type>& ftypes) : ftypes_(ftypes) {}

    feature_types_impl(const feature_types_impl&) = delete;
    feature_types_impl& operator=(const feature_types_impl&) = delete;

    std::int64_t get_feature_count() const;
    const array<feature_type>& get_array() const;

private:
    const array<feature_type> ftypes_;
};

class feature_types {
public:
    explicit feature_types(feature_types_impl* impl) : impl_(impl) {}
    explicit feature_types(const array<feature_type>& ftypes)
            : feature_types(new feature_types_impl{ ftypes }) {}

    std::int64_t get_feature_count() const;
    const array<feature_type>& get_array() const;

private:
    mutable dal::detail::pimpl<feature_types_impl> impl_;
};

class feature_types_builder_impl {
public:
    explicit feature_types_builder_impl(std::int64_t feature_count)
            : feature_count_(feature_count),
              ftypes_(array<feature_type>::full(feature_count, feature_type::ratio)) {}

    feature_types_builder_impl(const feature_types_builder_impl&) = delete;
    feature_types_builder_impl& operator=(const feature_types_builder_impl&) = delete;

    void set_default(feature_type type);
    void set(std::int64_t idx, feature_type type);
    void set(const range& r, feature_type type);

    feature_types build() const;

private:
    const std::int64_t feature_count_;
    array<feature_type> ftypes_;
};

class feature_types_builder {
public:
    explicit feature_types_builder(std::int64_t feature_count)
            : feature_types_builder(new feature_types_builder_impl{ feature_count }) {}
    explicit feature_types_builder(feature_types_builder_impl* impl) : impl_(impl) {}

    feature_types_builder& set_default(feature_type type);
    feature_types_builder& set(std::int64_t idx, feature_type type);
    feature_types_builder& set(const range& r, feature_type type);

    feature_types build() const;

private:
    mutable dal::detail::pimpl<feature_types_builder_impl> impl_;
};

} // namespace oneapi::dal::test::engine
