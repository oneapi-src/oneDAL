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

#include "oneapi/dal/data/table.hpp"

namespace oneapi::dal::detail {

class csv_table_reader_impl_iface {
public:
    virtual ~csv_table_reader_impl_iface() {}

    virtual table read(const char * file_name) = 0;
    virtual void set_delimiter(char delimiter) = 0;
    virtual void set_parse_header(bool parse_header) = 0;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual table read(sycl::queue& queue, const char * file_name) = 0;
#endif
};

template <typename Impl>
class csv_table_reader_impl_wrapper : public csv_table_reader_impl_iface, public base {
public:
    csv_table_reader_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)) {}

    virtual table read(const char * file_name) override {
        return impl_.read(file_name);
    }

    virtual void set_delimiter(char delimiter) override {
        impl_.set_delimiter(delimiter);
    }

    virtual void set_parse_header(bool parse_header) override {
        impl_.set_parse_header(parse_header);
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual table read(sycl::queue& queue,
                       const char * file_name) override {
        return impl_.read(queue, file_name);
    }
#endif

private:
    Impl impl_;
};

} // namespace oneapi::dal::detail
