/* file: multiclassclassifier_train_oneagainstone_kernel.h */
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
//  Declaration of template structs for One-Against-One method for Multi-class classifier
//  training algorithm for CSR input data.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__

#include "multi_class_classifier_model.h"
#include "service_sort.h"
#include "service_memory.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{
//Base class for binary classification subtask
template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTask
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubTask() {}

    services::Status getDataSubset(size_t nFeatures, size_t nVectors, int classIdxPositive, int classIdxNegative, const int * y, size_t & nRows)
    {
        nRows = 0;
        /* Prepare "positive" observations of the training subset */
        services::Status s = copyDataIntoSubtable(nFeatures, nVectors, classIdxPositive, 1, y, nRows);
        if (s) /* Prepare "negative" observations of the training subset */
            s = copyDataIntoSubtable(nFeatures, nVectors, classIdxNegative, -1, y, nRows);
        return s;
    }

    services::Status trainSimpleClassifier(size_t nRowsInSubset)
    {
        _subsetXTable->resize(nRowsInSubset);
        _subsetYTable->resize(nRowsInSubset);

        typename ClsType::InputType * input = _simpleTraining->getInput();
        DAAL_CHECK(input, services::ErrorNullInput);
        input->set(classifier::training::data, _subsetXTable);
        input->set(classifier::training::labels, _subsetYTable);
        services::Status s;
        DAAL_CHECK_STATUS(s, _simpleTraining->resetResult());
        return _simpleTraining->computeNoThrow();
    }

    classifier::ModelPtr getModel() { return _simpleTraining->getResult()->get(classifier::training::model); }

protected:
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;

    SubTask(size_t nSubsetVectors, size_t dataSize, const services::SharedPtr<ClsType> & st) : _subsetX(dataSize + nSubsetVectors), _subsetY(nullptr)
    {
        services::Status status;
        if (!_subsetX.get()) return;
        _subsetY      = _subsetX.get() + dataSize;
        _subsetYTable = HomogenNT::create(_subsetY, 1, nSubsetVectors, &status);
        if (!status) return;
        _simpleTraining = st->clone();
    }

    bool isValid() const { return _subsetX.get() && _subsetYTable.get() && _simpleTraining.get(); }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const int * y,
                                                  size_t & nRows) = 0;

protected:
    TArray<algorithmFPType, cpu> _subsetX;
    algorithmFPType * _subsetY;
    NumericTablePtr _subsetYTable;
    NumericTablePtr _subsetXTable;
    services::SharedPtr<ClsType> _simpleTraining;
};

template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTaskCSR : public SubTask<algorithmFPType, ClsType, cpu>
{
public:
    typedef SubTask<algorithmFPType, ClsType, cpu> super;
    static SubTaskCSR * create(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable,
                               const services::SharedPtr<ClsType> & st)
    {
        auto val = new SubTaskCSR(nFeatures, nSubsetVectors, dataSize, dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable)), st);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    bool isValid() const { return super::isValid() && _colIndicesX.get() && this->_subsetXTable.get(); }

    SubTaskCSR(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, CSRNumericTableIface * xTable, const services::SharedPtr<ClsType> & st)
        : super(nSubsetVectors, dataSize, st), _mtX(xTable), _colIndicesX(dataSize + nSubsetVectors + 1), _rowOffsetsX(nullptr)
    {
        if (_colIndicesX.get())
        {
            _rowOffsetsX = _colIndicesX.get() + dataSize;
            services::Status s;
            this->_subsetXTable = CSRNumericTable::create(this->_subsetX.get(), _colIndicesX.get(), _rowOffsetsX, nFeatures, 0,
                                                          CSRNumericTableIface::CSRIndexing::oneBased, &s);
            if (!s) return;
        }
    }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const int * y,
                                                  size_t & nRows) DAAL_C11_OVERRIDE;

private:
    TArray<size_t, cpu> _colIndicesX;
    size_t * _rowOffsetsX;
    ReadRowsCSR<algorithmFPType, cpu> _mtX;
};

template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTaskDense : public SubTask<algorithmFPType, ClsType, cpu>
{
public:
    typedef SubTask<algorithmFPType, ClsType, cpu> super;
    static SubTaskDense * create(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable,
                                 const services::SharedPtr<ClsType> & st)
    {
        auto val = new SubTaskDense(nFeatures, nSubsetVectors, dataSize, xTable, st);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    bool isValid() const { return super::isValid() && this->_subsetXTable.get(); }

    SubTaskDense(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable, const services::SharedPtr<ClsType> & st)
        : super(nSubsetVectors, dataSize, st), _mtX(const_cast<NumericTable *>(xTable))
    {
        services::Status status;
        if (this->_subsetX.get()) this->_subsetXTable = HomogenNT::create(this->_subsetX.get(), nFeatures, nSubsetVectors, &status);
        if (!status) return;
    }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const int * y,
                                                  size_t & nRows) DAAL_C11_OVERRIDE;

private:
    ReadRows<algorithmFPType, cpu> _mtX;
};

template <typename algorithmFPType, typename ClsType, typename MccParType, CpuType cpu>
class MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, ClsType, MccParType, cpu> : public Kernel
{
public:
    services::Status compute(const NumericTable * xTable, const NumericTable * yTable, daal::algorithms::Model * r,
                             const daal::algorithms::Parameter * par);

protected:
    services::Status computeDataSize(size_t nVectors, size_t nFeatures, size_t nClasses, const NumericTable * xTable, const int * y,
                                     size_t & nSubsetVectors, size_t & dataSize);
};

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
