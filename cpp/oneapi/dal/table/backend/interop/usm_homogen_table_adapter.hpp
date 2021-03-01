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

#include "oneapi/dal/table/backend/interop/homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

#ifdef ONEDAL_DATA_PARALLEL
// This class shall be used only to represent immutable data on DAAL side. Any
// attempts to change the data inside objects of that class lead to undefined
// behavior.
template <typename Data>
class usm_homogen_table_adapter : public homogen_table_adapter<Data> {
    using base = homogen_table_adapter<Data>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;
    using ptr_t = daal::services::SharedPtr<usm_homogen_table_adapter>;
    using ptr_data_t = daal::services::SharedPtr<Data>;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

    template <typename T>
    using daal_buffer_t = daal::services::internal::Buffer<T>;

    template <typename T>
    using daal_buffer_and_status_t = std::tuple<daal_buffer_t<T>, status_t>;

public:
    static ptr_t create(sycl::queue& q, const homogen_table& table);

private:
    explicit usm_homogen_table_adapter(sycl::queue& q, const homogen_table& table, status_t& stat);

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<double>& block) override;

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<float>& block) override;

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<int>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<double>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<float>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<int>& block) override;

    template <typename BlockData>
    status_t read_rows_impl(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<BlockData>& block);

    template <typename BlockData>
    status_t read_column_values_impl(std::size_t feature_idx,
                                     std::size_t vector_idx,
                                     std::size_t value_num,
                                     rw_mode_t rwflag,
                                     block_desc_t<BlockData>& block);

    template <typename BlockData>
    daal_buffer_and_status_t<BlockData> convert_to_daal_buffer(const array<BlockData>& ary) const;

    template <typename BlockData>
    daal_buffer_and_status_t<BlockData> pull_rows_buffer(const block_info& info);

    template <typename BlockData>
    daal_buffer_and_status_t<BlockData> pull_columns_buffer(const block_info& info);

    sycl::queue queue_;
};
#endif

} // namespace oneapi::dal::backend::interop
