/* file: svm_train_common.h */
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

#ifndef __SVM_TRAIN_COMMON_H__
#define __SVM_TRAIN_COMMON_H__

#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/service_utils.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::internal;

enum SVMVectorStatus
{
    free   = 0x0,
    up     = 0x1,
    low    = 0x2,
    shrink = 0x4
};

template <typename algorithmFPType, CpuType cpu>
struct HelperTrainSVM
{
    DAAL_FORCEINLINE static bool isUpper(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }
    DAAL_FORCEINLINE static bool isLower(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

    DAAL_FORCEINLINE static algorithmFPType WSSi(size_t nActiveVectors, const algorithmFPType * grad, const char * I, int & Bi);

    DAAL_FORCEINLINE static void WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                           const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                           const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                           algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta);

private:
    DAAL_FORCEINLINE static void WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                   const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                   const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                                   algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta);
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskBase
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubDataTaskBase() {}

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const NumericTablePtr & xTable) = 0;

    NumericTablePtr getTableData() const { return _dataTable; }

protected:
    SubDataTaskBase(const size_t nSubsetVectors, const size_t dataSize) : _nSubsetVectors(nSubsetVectors), _data(dataSize) {}
    SubDataTaskBase(const size_t nSubsetVectors) : _nSubsetVectors(nSubsetVectors) {}

    bool isValid() const { return _data.get(); }

protected:
    size_t _nSubsetVectors;
    TArray<algorithmFPType, cpu> _data;
    NumericTablePtr _dataTable;
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskCSR : public SubDataTaskBase<algorithmFPType, cpu>
{
public:
    using super = SubDataTaskBase<algorithmFPType, cpu>;
    static SubDataTaskCSR * create(const NumericTablePtr & xTable, const size_t nSubsetVectors)
    {
        auto val = new SubDataTaskCSR(xTable, nSubsetVectors);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    bool isValid() const { return super::isValid() && _colIndices.get() && this->_dataTable.get(); }

    SubDataTaskCSR(const NumericTablePtr & xTable, const size_t nSubsetVectors) : super(nSubsetVectors)
    {
        const size_t p                        = xTable->getNumberOfColumns();
        const size_t nRows                    = xTable->getNumberOfRows();
        CSRNumericTableIface * const csrIface = dynamic_cast<CSRNumericTableIface * const>(const_cast<NumericTable *>(xTable.get()));
        if (!csrIface) return;
        ReadRowsCSR<algorithmFPType, cpu> mtX(csrIface, 0, nRows);
        const size_t * const rowOffsets = mtX.rows();
        const size_t maxDataSize        = rowOffsets[nRows] - rowOffsets[0];
        this->_data.reset(maxDataSize);
        _colIndices.reset(maxDataSize + nSubsetVectors + 1);
        _rowOffsets = _colIndices.get() + maxDataSize;
        if (this->_data.get())
            this->_dataTable = CSRNumericTable::create(this->_data.get(), _colIndices.get(), _rowOffsets, p, nSubsetVectors,
                                                       CSRNumericTableIface::CSRIndexing::oneBased);
    }

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const NumericTablePtr & xTable);

private:
    TArray<size_t, cpu> _colIndices;
    size_t * _rowOffsets;
};

template <typename algorithmFPType, CpuType cpu>
class SubDataTaskDense : public SubDataTaskBase<algorithmFPType, cpu>
{
public:
    using super = SubDataTaskBase<algorithmFPType, cpu>;
    static SubDataTaskDense * create(const size_t nFeatures, const size_t nSubsetVectors)
    {
        auto val = new SubDataTaskDense(nFeatures, nSubsetVectors);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

    virtual services::Status copyDataByIndices(const uint32_t * wsIndices, const NumericTablePtr & xTable);

private:
    bool isValid() const { return super::isValid() && this->_dataTable.get(); }

    SubDataTaskDense(const size_t nFeatures, const size_t nSubsetVectors) : super(nSubsetVectors, nFeatures * nSubsetVectors)
    {
        services::Status status;
        if (this->_data.get())
            this->_dataTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(this->_data.get(), nFeatures, nSubsetVectors, &status);
    }
};

template <typename algorithmFPType, CpuType cpu>
using SubDataTaskBasePtr = services::SharedPtr<SubDataTaskBase<algorithmFPType, cpu> >;

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
