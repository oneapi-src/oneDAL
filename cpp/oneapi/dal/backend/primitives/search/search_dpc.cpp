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

#include "oneapi/dal/backend/primitives/common.hpp"

#include "oneapi/dal/backend/primitives/search/search.hpp"
#include "oneapi/dal/backend/primitives/search/common.hpp"

namespace oneapi::dal::backend::primitives {

template<typename Float, typename Distance>
sycl::event search_engine<Float, Distance>::operator() (ndview<Float, 2>& inp1,
                                                        ndview<Float, 2>& inp2,
                                                        event_vector& deps = {}) {
    const auto inp1_row_count = inp1.get_dimension();
    const auto inp2_row_count = inp2.get_dimension();
    const auto col_count = inp1.get_dimension();
    ONEDAL_ASSERT(col_count == inp2.get_dimension());
    const auto block_count1 = get_block_count(inp1_row_count, block_size1_);
    const auto block_count2 = get_block_count(inp2_row_count, block_size2_);
    for(std::int64_t i = 0; i < block1_count; ++i) {
        const auto block1_size = get_block_size(i, 
                                                inp1_row_count, 
                                                block_size1_);
        const auto block1 = get_block()
        for(std::int64_t j = 0; j < block2_count; ++j) {
            const auto block2_size = get_block_size(j, 
                                                    inp2_row_count, 
                                                    block_size2_);
            const auto dist_event = dist_();
            

        }
    }
}


} // oneapi::dal::backend::primitives
