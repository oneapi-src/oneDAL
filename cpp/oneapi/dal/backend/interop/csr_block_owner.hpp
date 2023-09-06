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

namespace oneapi::dal::backend::interop {

template <typename T>
struct csr_block_owner {
    csr_block_owner(const daal::data_management::NumericTablePtr nt) {
        _is_empty = false;
        daal::services::Status status;
        _csr_nt = daal::services::dynamicPointerCast<daal::data_management::CSRNumericTable,
                                                     daal::data_management::NumericTable>(nt);
        ONEDAL_ASSERT(_csr_nt);
        _row_count = dal::detail::integral_cast<std::int64_t>(_csr_nt->getNumberOfRows());
        _column_count = dal::detail::integral_cast<std::int64_t>(_csr_nt->getNumberOfColumns());
        _element_count = dal::detail::integral_cast<std::int64_t>(_csr_nt->getDataSize());
        status = _csr_nt->getSparseBlock(0,
                                         _csr_nt->getNumberOfRows(),
                                         daal::data_management::readOnly,
                                         _block);
        status_to_exception(status);
    }

    ~csr_block_owner() noexcept(false) {
        if (!_is_empty) {
            daal::services::Status status = _csr_nt->releaseSparseBlock(_block);
            status_to_exception(status);
            _is_empty = true;
        }
    }

    std::int64_t get_row_count() const {
        return _row_count;
    }

    std::int64_t get_column_count() const {
        return _column_count;
    }

    std::int64_t get_element_count() const {
        return _element_count;
    }

    T* get_data() const {
        return _block.getBlockValuesPtr();
    }

    std::size_t* get_column_indices() const {
        return _block.getBlockColumnIndicesPtr();
    }

    std::size_t* get_row_indices() const {
        return _block.getBlockRowIndicesPtr();
    }

    daal::data_management::CSRBlockDescriptor<T> _block;
    daal::data_management::CSRNumericTablePtr _csr_nt;
    std::int64_t _row_count;
    std::int64_t _column_count;
    std::int64_t _element_count;
    bool _is_empty;
};

} // namespace oneapi::dal::backend::interop
