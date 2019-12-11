/* file: service_numeric_table.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/*
//++
//  CPU-specified homogeneous numeric table
//--
*/

#ifndef __SERVICE_NUMERIC_TABLE_H__
#define __SERVICE_NUMERIC_TABLE_H__

#include "homogen_numeric_table.h"
#include "csr_numeric_table.h"
#include "symmetric_matrix.h"
#include "service_defines.h"
#include "service_memory.h"
#include "service_arrays.h"

using namespace daal::data_management;

#define DEFINE_TABLE_BLOCK_EX(BlockType, targetVariable, ...)    \
    BlockType<algorithmFPType, cpu> targetVariable(__VA_ARGS__); \
    DAAL_CHECK_BLOCK_STATUS(targetVariable);

#define DEFINE_TABLE_BLOCK(BlockType, targetVariable, table) DEFINE_TABLE_BLOCK_EX(BlockType, targetVariable, table, 0, (table)->getNumberOfRows())

namespace daal
{
namespace internal
{
template <CpuType cpu>
class NumericTableFeatureCPU : public NumericTableFeature
{
public:
    NumericTableFeatureCPU() : NumericTableFeature() {}
    virtual ~NumericTableFeatureCPU() {}
};

template <CpuType cpu>
class NumericTableDictionaryCPU : public NumericTableDictionary
{
public:
    NumericTableDictionaryCPU(size_t nfeat)
    {
        _nfeat         = 0;
        _dict          = (NumericTableFeature *)(new NumericTableFeatureCPU<cpu>[1]);
        _featuresEqual = DictionaryIface::equal;
        if (nfeat) setNumberOfFeatures(nfeat);
    };

    services::Status setAllFeatures(const NumericTableFeature & defaultFeature) DAAL_C11_OVERRIDE
    {
        if (_nfeat > 0)
        {
            _dict[0] = defaultFeature;
        }
        return services::Status();
    }

    services::Status setNumberOfFeatures(size_t nfeat) DAAL_C11_OVERRIDE
    {
        _nfeat = nfeat;
        return services::Status();
    }
};

template <typename T, CpuType cpu>
class HomogenNumericTableCPU
{};

template <CpuType cpu>
class HomogenNumericTableCPU<float, cpu> : public HomogenNumericTable<float>
{
public:
    HomogenNumericTableCPU(float * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<float>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;

        NumericTableFeature df;
        df.setType<float>();
        st.add(_ddict->setAllFeatures(df));

        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<float, cpu> > create(float * const ptr, size_t featnum, size_t obsnum,
                                                                           services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(float, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<float>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        st.add(setNumberOfRows(obsnum));

        NumericTableFeature df;
        df.setType<float>();
        st.add(_ddict->setAllFeatures(df));

        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<float, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(float, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }
};

template <CpuType cpu>
class HomogenNumericTableCPU<double, cpu> : public HomogenNumericTable<double>
{
public:
    HomogenNumericTableCPU(double * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<double>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        NumericTableFeature df;
        df.setType<double>();
        st.add(_ddict->setAllFeatures(df));

        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<double, cpu> > create(double * const ptr, size_t featnum, size_t obsnum,
                                                                            services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(double, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<double>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st)
    {
        st.add(setNumberOfRows(obsnum));

        NumericTableFeature df;
        df.setType<double>();
        st.add(_ddict->setAllFeatures(df));

        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<double, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(double, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }
};

template <CpuType cpu>
class HomogenNumericTableCPU<int, cpu> : public HomogenNumericTable<int>
{
public:
    HomogenNumericTableCPU(int * const ptr, size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<int>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st), _cpuDict(nullptr)
    {
        NumericTableFeature df;
        df.setType<int>();
        st.add(_ddict->setAllFeatures(df));

        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        st.add(setNumberOfRows(obsnum));
    }

    static services::SharedPtr<HomogenNumericTableCPU<int, cpu> > create(int * const ptr, size_t featnum, size_t obsnum,
                                                                         services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(int, cpu), ptr, featnum, obsnum);
    }

    HomogenNumericTableCPU(size_t featnum, size_t obsnum, services::Status & st)
        : HomogenNumericTable<int>(services::SharedPtr<NumericTableDictionaryCPU<cpu> >(new NumericTableDictionaryCPU<cpu>(featnum)), st), _cpuDict(nullptr)
    {
        NumericTableFeature df;
        df.setType<int>();
        st.add(_ddict->setAllFeatures(df));

        st.add(setNumberOfRows(obsnum));
        st.add(allocateDataMemory());
    }

    static services::SharedPtr<HomogenNumericTableCPU<int, cpu> > create(size_t featnum, size_t obsnum, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTableCPU, DAAL_TEMPLATE_ARGUMENTS(int, cpu), featnum, obsnum);
    }

    virtual ~HomogenNumericTableCPU() {}

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

private:
    NumericTableDictionary * _cpuDict;
};

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetRows
{
public:
    GetRows(NumericTableType & data, size_t iStartFrom, size_t nRows) : _data(&data) { getBlock(iStartFrom, nRows); }
    GetRows(NumericTableType * data, size_t iStartFrom, size_t nRows) : _data(data), _toReleaseFlag(false)
    {
        if (_data)
        {
            getBlock(iStartFrom, nRows);
        }
    }
    GetRows(NumericTableType * data = nullptr) : _data(data), _toReleaseFlag(false) {}
    GetRows(NumericTableType & data) : _data(&data), _toReleaseFlag(false) {}
    ~GetRows() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * next(size_t iStartFrom, size_t nRows)
    {
        if (!_data) return nullptr;

        if (_toReleaseFlag)
        {
            _status = _data->releaseBlockOfRows(_block);
        }

        return getBlock(iStartFrom, nRows);
    }
    algorithmFPAccessType * set(NumericTableType * data, size_t iStartFrom, size_t nRows)
    {
        release();
        if (data)
        {
            _data = data;
            return getBlock(iStartFrom, nRows);
        }
        return nullptr;
    }
    algorithmFPAccessType * set(NumericTableType & data, size_t iStartFrom, size_t nRows)
    {
        release();
        _data = &data;
        return getBlock(iStartFrom, nRows);
    }
    void release()
    {
        if (_toReleaseFlag)
        {
            _data->releaseBlockOfRows(_block);
            _toReleaseFlag = false;
        }
        _data = nullptr;
        _status.clear();
    }
    const services::Status & status() const { return _status; }

private:
    algorithmFPAccessType * getBlock(size_t iStartFrom, size_t nRows)
    {
        _status        = _data->getBlockOfRows(iStartFrom, nRows, mode, _block);
        _toReleaseFlag = _status.ok();
        return _block.getBlockPtr();
    }

private:
    NumericTableType * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
    bool _toReleaseFlag;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadRows = GetRows<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteRows = GetRows<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyRows = GetRows<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode>
class GetRowsCSR
{
public:
    GetRowsCSR(CSRNumericTableIface & data, size_t iStartFrom, size_t nRows) : _data(&data) { getBlock(iStartFrom, nRows); }
    GetRowsCSR(CSRNumericTableIface * data, size_t iStartFrom, size_t nRows) : _data(data), _toReleaseFlag(false)
    {
        if (_data)
        {
            getBlock(iStartFrom, nRows);
        }
    }
    GetRowsCSR(CSRNumericTableIface * data = nullptr) : _data(data), _toReleaseFlag(false) {}
    ~GetRowsCSR() { release(); }

    const algorithmFPAccessType * values() const { return _data ? _block.getBlockValuesPtr() : nullptr; }
    const size_t * cols() const { return _data ? _block.getBlockColumnIndicesPtr() : nullptr; }
    const size_t * rows() const { return _data ? _block.getBlockRowIndicesPtr() : nullptr; }
    algorithmFPAccessType * values() { return _data ? _block.getBlockValuesPtr() : nullptr; }
    size_t * cols() { return _data ? _block.getBlockColumnIndicesPtr() : nullptr; }
    size_t * rows() { return _data ? _block.getBlockRowIndicesPtr() : nullptr; }

    void next(size_t iStartFrom, size_t nRows)
    {
        if (_data)
        {
            if (_toReleaseFlag)
            {
                _status = _data->releaseSparseBlock(_block);
            }
            getBlock(iStartFrom, nRows);
        }
    }
    void set(CSRNumericTableIface * data, size_t iStartFrom, size_t nRows)
    {
        release();
        if (data)
        {
            _data = data;
            getBlock(iStartFrom, nRows);
        }
    }
    void release()
    {
        if (_toReleaseFlag)
        {
            _data->releaseSparseBlock(_block);
            _toReleaseFlag = false;
        }
        _data = nullptr;
        _status.clear();
    }

    const services::Status & status() const { return _status; }
    size_t size() { return _block.getDataSize(); }

private:
    void getBlock(size_t iStartFrom, size_t nRows)
    {
        _status        = _data->getSparseBlock(iStartFrom, nRows, mode, _block);
        _toReleaseFlag = _status.ok();
    }

private:
    CSRNumericTableIface * _data;
    CSRBlockDescriptor<algorithmFPType> _block;
    services::Status _status;
    bool _toReleaseFlag;
};

template <typename algorithmFPType, CpuType cpu>
using ReadRowsCSR = GetRowsCSR<algorithmFPType, const algorithmFPType, cpu, readOnly>;

template <typename algorithmFPType, CpuType cpu>
using WriteRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, readWrite>;

template <typename algorithmFPType, CpuType cpu>
using WriteOnlyRowsCSR = GetRowsCSR<algorithmFPType, algorithmFPType, cpu, writeOnly>;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetColumns
{
public:
    GetColumns(NumericTableType & data, size_t iCol, size_t iStartFrom, size_t n) : _data(&data)
    {
        _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
    }
    GetColumns(NumericTableType * data, size_t iCol, size_t iStartFrom, size_t n) : _data(data)
    {
        if (_data) _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
    }
    GetColumns() : _data(nullptr) {}
    ~GetColumns() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * next(size_t iCol, size_t iStartFrom, size_t n)
    {
        if (!_data) return nullptr;
        _status = _data->releaseBlockOfColumnValues(_block);
        _status |= _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
        return _block.getBlockPtr();
    }
    algorithmFPAccessType * set(NumericTableType * data, size_t iCol, size_t iStartFrom, size_t n)
    {
        release();
        if (data)
        {
            _data   = data;
            _status = _data->getBlockOfColumnValues(iCol, iStartFrom, n, mode, _block);
            return _block.getBlockPtr();
        }
        return nullptr;
    }
    void release()
    {
        if (_data)
        {
            _data->releaseBlockOfColumnValues(_block);
            _data = nullptr;
            _status.clear();
        }
    }
    const services::Status & status() const { return _status; }

private:
    NumericTableType * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadColumns = GetColumns<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteColumns = GetColumns<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyColumns = GetColumns<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

template <typename algorithmFPType, typename algorithmFPAccessType, CpuType cpu, ReadWriteMode mode, typename NumericTableType>
class GetPacked
{
public:
    GetPacked(NumericTableType & data)
    {
        _data = dynamic_cast<PackedArrayNumericTableIface *>(&data);
        if (!_data)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return;
        }
        _status = _data->getPackedArray(mode, _block);
    }
    GetPacked(NumericTableType * data)
    {
        _data = dynamic_cast<PackedArrayNumericTableIface *>(data);
        if (!_data)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return;
        }
        _status = _data->getPackedArray(mode, _block);
    }
    GetPacked() : _data(nullptr) {}
    ~GetPacked() { release(); }
    algorithmFPAccessType * get() { return _data ? _block.getBlockPtr() : nullptr; }
    algorithmFPAccessType * set(NumericTableType * data)
    {
        release();
        PackedArrayNumericTableIface * ptr = dynamic_cast<PackedArrayNumericTableIface *>(data);
        if (!ptr)
        {
            _status = services::Status(services::ErrorIncorrectTypeOfNumericTable);
            return nullptr;
        }
        if (ptr)
        {
            _data   = ptr;
            _status = _data->getPackedArray(mode, _block);
            return _block.getBlockPtr();
        }
        return nullptr;
    }
    void release()
    {
        if (_data)
        {
            _data->releasePackedArray(_block);
            _data = nullptr;
            _status.clear();
        }
    }
    const services::Status & status() const { return _status; }

private:
    PackedArrayNumericTableIface * _data;
    BlockDescriptor<algorithmFPType> _block;
    services::Status _status;
};

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using ReadPacked = GetPacked<algorithmFPType, const algorithmFPType, cpu, readOnly, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WritePacked = GetPacked<algorithmFPType, algorithmFPType, cpu, readWrite, NumericTableType>;

template <typename algorithmFPType, CpuType cpu, typename NumericTableType = NumericTable>
using WriteOnlyPacked = GetPacked<algorithmFPType, algorithmFPType, cpu, writeOnly, NumericTableType>;

using daal::services::internal::TArray;
using daal::services::internal::TArrayCalloc;
using daal::services::internal::TArrayScalable;
using daal::services::internal::TArrayScalableCalloc;

using daal::services::internal::TNArray;

template <typename algorithmFPType>
services::Status createSparseTable(const NumericTablePtr & inputTable, CSRNumericTablePtr & resTable);

} // namespace internal
} // namespace daal

#endif
