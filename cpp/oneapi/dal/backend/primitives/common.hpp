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

#include "oneapi/dal/backend/common.hpp"

class uniform_blocking {
public:
    uniform_blocking(std::int64_t length, std::int64_t block) 
        : range_length_{ length }, block_length_{ block } {
        ONEDAL_ASSERT(block > 0);
    }
    
    const std::int64_t& block() const {
        return block_length_;
    }

    const std::int64_t& length() const {
        return range_length_;
    }

    std::int64_t block_count() const {
        return (length() / block())  + bool(length() % block());
    }

    std::int64_t block_begin(std::int64_t i) const {
        ONEDAL_ASSERT((block_count() > i) && (i >= 0));
        return i * block();
    }

    std::int64_t block_end(std::int64_t i) const {
        ONEDAL_ASSERT((block_count() > i) && (i >= 0));
        return std::min((i + 1l) * block(), length());
    }

private:
    const std::int64_t range_length_;
    const std::int64_t block_length_;
};
