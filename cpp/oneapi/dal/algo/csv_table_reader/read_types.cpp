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

#include "oneapi/dal/algo/csv_table_reader/read_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/data/table.hpp"

namespace oneapi::dal::csv_table_reader {

template<>
class detail::read_input_impl<table> : public base {
public:
    read_input_impl(const char * file_name) {
        const size_t len = strlen(file_name);
        this->file_name = new char[len + 1];
        dal::detail::memcpy(dal::detail::default_host_policy{}, this->file_name, file_name, sizeof(char) * len);
        this->file_name[len] = '\0';
    }

    ~read_input_impl() {
        delete [] file_name;
    }

    char * file_name;
};

template<>
class detail::read_result_impl<table> : public base {
public:
    table value;
};

read_input<table>::read_input(const char * file_name) : impl_(new detail::read_input_impl<table>(file_name)) {}

const char * read_input<table>::get_file_name() const {
    return impl_->file_name;
}

void read_input<table>::set_file_name_impl(const char * file_name) {
    const size_t len = strlen(file_name);
    impl_->file_name = new char[len + 1];
    dal::detail::memcpy(dal::detail::default_host_policy{}, impl_->file_name, file_name, sizeof(char) * len);
    impl_->file_name[len] = '\0';
}

read_result<table>::read_result() : impl_(new detail::read_result_impl<table>{}) {}

table read_result<table>::get_table() const {
    return impl_->value;
}

void read_result<table>::set_table_impl(const table& value) {
    impl_->value = value;
}

} // namespace oneapi::dal::pca
