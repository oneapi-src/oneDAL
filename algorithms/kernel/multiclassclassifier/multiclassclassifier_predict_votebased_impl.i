/* file: multiclassclassifier_predict_votebased_impl.i */
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
//  Implementation of score-based method for Multi-class classifier
//  prediction algorithm.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_VOTEBASED_IMPL_I__
#define __MULTICLASSCLASSIFIER_PREDICT_VOTEBASED_IMPL_I__

#include "multi_class_classifier_model.h"
#include "threading.h"
#include "service_error_handling.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, typename ClsType, typename MultiClsParam, CpuType cpu>
struct MultiClassClassifierPredictKernel<voteBased, training::oneAgainstOne, algorithmFPType, ClsType, MultiClsParam, cpu> : public Kernel
{
    Status compute(const NumericTable * a, const daal::algorithms::Model * m, NumericTable * r, const daal::algorithms::Parameter * par);
};

/** Base class for threading subtask */
template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTaskVoteBased
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubTaskVoteBased() {}

    /**
     * Computes a block of predictions
     * \param[in] startRow  Index of the starting row in the block
     * \param[in] nRows     Number of rows in the block
     * \param[in] a         Numeric table of size n x p with input data set
     * \param[in] model     Model of the multi-class classifier
     * \param[out] r        Numeric table of size n x 1 with resulting labels
     * \param[in] nonEmptyClassMap Array that contains indices of non-empty classes
     * \return Status of the computations
     */
    Status predict(size_t startRow, size_t nRows, const NumericTable * a, Model * model, NumericTable * r, const size_t * nonEmptyClassMap)
    {
        Status s;
        algorithmFPType * y = _aY.get();     // array of two-class classifier predictions
        int * votes         = _aVotes.get(); // array of two-class classifiers' votes

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses * nRows; i++)
        {
            votes[i] = 0;
        }

        {
            NumericTablePtr xTable; // block of rows from the input data set
            s = getDataBlock(startRow, nRows, a, xTable);
            if (!s) return s;

            if (nRows != _yTable->getNumberOfRows()) _yTable->resize(nRows);

            /* TODO: Try threading here */
            for (size_t iClass = 1, imodel = 0; iClass < _nClasses; iClass++)
            {
                for (size_t jClass = 0; jClass < iClass; jClass++, imodel++)
                {
                    /* Compute two-class predictions for the pair of classes (iClass, jClass)
                       for the block of input observations */
                    classifier::prediction::Input * input = _simplePrediction->getInput();
                    DAAL_CHECK(input, ErrorNullInput);
                    input->set(classifier::prediction::data, xTable);
                    input->set(classifier::prediction::model, model->getTwoClassClassifierModel(imodel));

                    s = _simplePrediction->computeNoThrow();
                    if (!s) return Status(ErrorMultiClassFailedToComputeTwoClassPrediction).add(s);

                    /* Compute votes for the block of input observations */
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < nRows; i++)
                    {
                        if (y[i] >= 0)
                            votes[i * _nClasses + iClass]++;
                        else
                            votes[i * _nClasses + jClass]++;
                    }
                }
            }
        }

        /* Compute resulting labels as indices of the maximum vote values */
        WriteOnlyRows<int, cpu> res(r, startRow, nRows);
        int * labels = res.get();
        DAAL_CHECK_MALLOC(labels);

        int * votesPtr = votes;
        for (size_t i = 0; i < nRows; i++, votesPtr += _nClasses)
        {
            labels[i]   = nonEmptyClassMap[0];
            int maxVote = votesPtr[0];
            for (size_t iClass = 1; iClass < _nClasses; iClass++)
            {
                if (votesPtr[iClass] > maxVote)
                {
                    maxVote   = votesPtr[iClass];
                    labels[i] = nonEmptyClassMap[iClass];
                }
            }
        }
        return s;
    }

protected:
    /**
     * Get block of input observations
     * \param[in] startRow  Index of the starting row in the block
     * \param[in] nRows     Number of rows in the block
     * \param[in] a         Numeric table of size n x p with input data set
     * \param[out] xTable   Numeric table of size nRows x p with the block of observations
     * \return Status of the computations
     */
    virtual Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) = 0;

    /**
     * Constructs a base threading subtask
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \param[out] valid            Flag. True if the task was constructed successfully, false otherwise
     */
    SubTaskVoteBased(size_t nClasses, size_t nRows, const SharedPtr<ClsType> & simplePrediction, bool & valid)
        : _nClasses(nClasses),
          _aY(nRows),
          _aVotes(nClasses * nRows),
          _yRes(new typename ClsType::ResultType()),
          _simplePrediction(simplePrediction->clone())
    {
        if (!_aY.get() || !_aVotes.get() || !_yRes)
        {
            valid = false;
            return;
        }

        _yTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_aY.get(), 1, nRows);
        if (!_yTable)
        {
            valid = false;
            return;
        }
        _yRes->set(classifier::prediction::prediction, _yTable);
        _simplePrediction->setResult(_yRes);
    }

private:
    size_t _nClasses;
    TArray<algorithmFPType, cpu> _aY;
    TArray<int, cpu> _aVotes;
    NumericTablePtr _yTable;
    services::SharedPtr<typename ClsType::ResultType> _yRes;
    SharedPtr<ClsType> _simplePrediction;
};

/** Class for threading subtask that works with dense input data */
template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTaskVoteBasedDense : public SubTaskVoteBased<algorithmFPType, ClsType, cpu>
{
public:
    typedef SubTaskVoteBased<algorithmFPType, ClsType, cpu> super;
    using super::predict;

    /**
     * Constructs a threading subtask that works with dense input data
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \return Pointer to the newly constructed subtask in case of success; NULL pointer in case of failure
     */
    static super * create(size_t nClasses, size_t nRows, const SharedPtr<ClsType> & simplePrediction)
    {
        bool valid  = true;
        super * res = new SubTaskVoteBasedDense(nClasses, nRows, simplePrediction, valid);
        if (res && valid) return res;
        delete res;
        return nullptr;
    }

protected:
    Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) DAAL_C11_OVERRIDE
    {
        _xRows.set(const_cast<NumericTable *>(a), startRow, nRows);
        DAAL_CHECK_BLOCK_STATUS(_xRows);
        Status s;
        xTable =
            HomogenNumericTableCPU<algorithmFPType, cpu>::create(const_cast<algorithmFPType *>(_xRows.get()), a->getNumberOfColumns(), nRows, &s);
        return s;
    }

private:
    SubTaskVoteBasedDense(size_t nClasses, size_t nRows, const SharedPtr<ClsType> & simplePrediction, bool & valid)
        : super(nClasses, nRows, simplePrediction, valid)
    {}
    ReadRows<algorithmFPType, cpu> _xRows;
};

/** Class for threading subtask that works with CSR input data */
template <typename algorithmFPType, typename ClsType, CpuType cpu>
class SubTaskVoteBasedCSR : public SubTaskVoteBased<algorithmFPType, ClsType, cpu>
{
public:
    typedef SubTaskVoteBased<algorithmFPType, ClsType, cpu> super;
    using super::predict;

    /**
     * Constructs a threading subtask that works with CSR input data
     * \param[in] nClasses          Number of classes
     * \param[in] nRows             Maximum number of rows processed in the iteration of a threader_for loop
     * \param[in] simplePrediction  Two-class classifier prediction algorithm
     * \return Pointer to the newly constructed subtask in case of success; NULL pointer in case of failure
     */
    static super * create(size_t nClasses, size_t nRows, const SharedPtr<ClsType> & simplePrediction)
    {
        bool valid  = true;
        super * res = new SubTaskVoteBasedCSR(nClasses, nRows, simplePrediction, valid);
        if (res && valid) return res;
        delete res;
        res = nullptr;
        return nullptr;
    }

protected:
    Status getDataBlock(size_t startRow, size_t nRows, const NumericTable * a, NumericTablePtr & xTable) DAAL_C11_OVERRIDE
    {
        _xRows.set(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a)), startRow, nRows);
        DAAL_CHECK_BLOCK_STATUS(_xRows);
        Status s;
        xTable = CSRNumericTable::create(const_cast<algorithmFPType *>(_xRows.values()), _xRows.rows(), _xRows.cols(), a->getNumberOfColumns(), nRows,
                                         CSRNumericTableIface::oneBased, &s);
        return s;
    }

private:
    SubTaskVoteBasedCSR(size_t nClasses, size_t nRows, const SharedPtr<ClsType> & simplePrediction, bool & valid)
        : super(nClasses, nRows, simplePrediction, valid)
    {}
    ReadRowsCSR<algorithmFPType, cpu> _xRows;
};

template <typename algorithmFPType, typename ClsType, typename MultiClsParam, CpuType cpu>
Status MultiClassClassifierPredictKernel<voteBased, training::oneAgainstOne, algorithmFPType, ClsType, MultiClsParam, cpu>::compute(
    const NumericTable * a, const daal::algorithms::Model * m, NumericTable * r, const daal::algorithms::Parameter * par)
{
    Model * model          = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
    MultiClsParam * mccPar = static_cast<MultiClsParam *>(const_cast<daal::algorithms::Parameter *>(par));
    size_t nClasses        = mccPar->nClasses;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClasses, sizeof(size_t));

    TArray<size_t, cpu> nonEmptyClassMapBuffer(nClasses);
    DAAL_CHECK_MALLOC(nonEmptyClassMapBuffer.get());

    size_t * nonEmptyClassMap = (size_t *)nonEmptyClassMapBuffer.get();
    Status s                  = getNonEmptyClassMap<algorithmFPType, cpu>(nClasses, model, nonEmptyClassMap);
    DAAL_CHECK_STATUS_VAR(s);

    const size_t nVectors = a->getNumberOfRows();

    SharedPtr<ClsType> simplePrediction = mccPar->prediction;

    const size_t nRowsInBlock = 256;
    size_t nBlocks            = nVectors / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nVectors) nBlocks++;

    typedef SubTaskVoteBased<algorithmFPType, ClsType, cpu> TSubTask;
    daal::ls<TSubTask *> lsTask([=, &simplePrediction]() {
        if (a->getDataLayout() == NumericTableIface::csrArray)
            return SubTaskVoteBasedCSR<algorithmFPType, ClsType, cpu>::create(nClasses, nRowsInBlock, simplePrediction);
        return SubTaskVoteBasedDense<algorithmFPType, ClsType, cpu>::create(nClasses, nRowsInBlock, simplePrediction);
    });

    /* Process input data set block by block */
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        TSubTask * local = lsTask.local();
        if (!local)
        {
            safeStat.add(ErrorMemoryAllocationFailed);
            return;
        }
        DAAL_LS_RELEASE(TSubTask, lsTask, local); //releases local storage when leaving this scope

        const size_t startRow = iBlock * nRowsInBlock;
        const size_t nRows    = (startRow + nRowsInBlock > nVectors) ? nVectors - startRow : nRowsInBlock;

        /* Get a block of predictions */
        Status s = local->predict(startRow, nRows, a, model, r, nonEmptyClassMap);
        DAAL_CHECK_STATUS_THR(s);
    });

    lsTask.reduce([=, &safeStat](TSubTask * local) { delete local; });
    return safeStat.detach();
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
