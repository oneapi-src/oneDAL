/* file: multiclassclassifier_predict_mccwu_kernel.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Declaration of template structs for Wu method for Multi-class classifier
//  prediction algorithm.
//--
*/
/*
//  REFERENCES
//
//  1. Ting-Fan Wu, Chih-Jen Lin, Ruby C. Weng
//     Probability Estimates for Multi-class Classification by Pairwise Coupling,
//     Journal of Machine Learning Research 5, 2004.
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_MCCWU_KERNEL_H__
#define __MULTICLASSCLASSIFIER_PREDICT_MCCWU_KERNEL_H__

#include "multi_class_classifier_model.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;

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

//Base class for binary classification subtask
template<typename algorithmFPType, CpuType cpu>
class SubTask
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubTask() {}

    /* Get multiclass classification results for a block of input observations */
    services::Status getBlockOfRowsOfResults(NumericTable *r, size_t nFeatures, size_t startRow, size_t nRows,
        size_t nClasses, const size_t *nonEmptyClassMap, Model *model, size_t nIter, double eps);

protected:
    SubTask(size_t nClasses, size_t nRowsInBlock, NumericTable *rTable,
        const services::SharedPtr<classifier::prediction::Batch>& sp) :
        _simplePrediction(sp->clone())
    {
        const size_t bufferSize = nRowsInBlock * nClasses * nClasses + nClasses * nClasses + 2 * nClasses + nRowsInBlock;
        _buffer.reset(bufferSize);
    }

    bool isValid() const
    {
        return _buffer.get() && _simplePrediction.get();
    }

    services::Status predictSimpleClassifier(size_t nFeatures, size_t startRow, size_t nRows,
        size_t nClasses, const size_t *nonEmptyClassMap, algorithmFPType *y, Model *model,
        algorithmFPType *rProb, const NumericTablePtr& xTable);

    /** Get 2-class classification probabilities for a block of observations */
    services::Status get2ClassProbabilities(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
        const size_t *nonEmptyClassMap, algorithmFPType *y,
        Model *model, algorithmFPType *rProb);

    virtual services::Status getInput(size_t nFeatures, size_t startRow, size_t nRows, NumericTablePtr& res) = 0;

protected:
    WriteOnlyColumns<int, cpu> _mtR;
    services::SharedPtr<classifier::prediction::Batch> _simplePrediction;
    TArray<algorithmFPType, cpu> _buffer;
};

template<typename algorithmFPType, CpuType cpu>
class SubTaskCSR : public SubTask<algorithmFPType, cpu>
{
public:
    typedef SubTask<algorithmFPType, cpu> super;
    static SubTaskCSR* create(size_t nClasses, size_t nRowsInBlock, const NumericTable *xTable, NumericTable *rTable,
        const services::SharedPtr<classifier::prediction::Batch>& sp)
    {
        auto val = new SubTaskCSR(nClasses, nRowsInBlock, dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable)), rTable, sp);
        if(val && val->isValid())
            return val;
        delete val;
        return nullptr;
    }

private:
    SubTaskCSR(size_t nClasses, size_t nRowsInBlock, CSRNumericTableIface *xTable, NumericTable *rTable,
        const services::SharedPtr<classifier::prediction::Batch>& sp) :
        super(nClasses, nRowsInBlock, rTable, sp), _mtX(xTable)
    {
    }
    virtual services::Status getInput(size_t nFeatures, size_t startRow, size_t nRows, NumericTablePtr& res) DAAL_C11_OVERRIDE;

private:
    ReadRowsCSR<algorithmFPType, cpu> _mtX;
};

template<typename algorithmFPType, CpuType cpu>
class SubTaskDense : public SubTask<algorithmFPType, cpu>
{
public:
    typedef SubTask<algorithmFPType, cpu> super;
    static SubTaskDense* create(size_t nClasses, size_t nRowsInBlock, const NumericTable *xTable, NumericTable *rTable,
        const services::SharedPtr<classifier::prediction::Batch>& sp)
    {
        auto val = new SubTaskDense(nClasses, nRowsInBlock, xTable, rTable, sp);
        if(val && val->isValid())
            return val;
        delete val;
        return nullptr;
    }

private:
    SubTaskDense(size_t nClasses, size_t nRowsInBlock, const NumericTable *xTable, NumericTable *rTable,
        const services::SharedPtr<classifier::prediction::Batch>& sp) :
        super(nClasses, nRowsInBlock, rTable, sp), _mtX(const_cast<NumericTable *>(xTable))
    {
    }
    virtual services::Status getInput(size_t nFeatures, size_t startRow, size_t nRows, NumericTablePtr& res) DAAL_C11_OVERRIDE;

private:
    ReadRows<algorithmFPType, cpu> _mtX;
};

template<typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>
        : public Kernel
{
    services::Status compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r,
                             const daal::algorithms::Parameter *par);
};

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
