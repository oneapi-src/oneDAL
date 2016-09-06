/* file: multiclassclassifier_predict_mccwu_kernel.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "threading.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_micro_table.h"
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

template<typename algorithmFPType, CpuType cpu>
size_t getMultiClassClassifierPredictBlockSize()
{
    return 128;
}

template<typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierTls
{
    MultiClassClassifierTls(size_t nClasses, size_t nRowsInBlock, const NumericTable *xTable, NumericTable *rTable,
                            services::SharedPtr<classifier::prediction::Batch> simplePrediction) :
        mtR(rTable),
        simplePrediction(simplePrediction->clone())
    {
        size_t bufferSize = nRowsInBlock * nClasses * nClasses + nClasses * nClasses + 2 * nClasses + nRowsInBlock;
        buffer = (algorithmFPType *)daal::services::daal_malloc(bufferSize * sizeof(algorithmFPType));
        if (!buffer) { error.setId(services::ErrorMemoryAllocationFailed); return; }
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            mtX = new CSRBlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
        }
        else
        {
            mtX = new BlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
        }
        if (!mtX) { error.setId(services::ErrorMemoryAllocationFailed); return; }
    }

    virtual ~MultiClassClassifierTls()
    {
        daal::services::daal_free(buffer);
        delete mtX;
    }

    MicroTable *mtX;
    FeatureMicroTable <int, writeOnly, cpu> mtR;
    services::SharedPtr<classifier::prediction::Batch> simplePrediction;
    algorithmFPType *buffer;
    services::Error error;
};

template<typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel<multiClassClassifierWu, training::oneAgainstOne, algorithmFPType, cpu>
        : public Kernel
{
    void compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r,
                 const daal::algorithms::Parameter *par);

protected:
    /* Get multiclass classification results for a block of input observations */
    inline void getBlockOfRowsOfResults(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                                        MicroTable *mtX, FeatureMicroTable<int, writeOnly, cpu> &mtR,
                                        services::SharedPtr<classifier::prediction::Batch> simplePrediction,
                                        Model *model,
                                        size_t nIter, double eps, algorithmFPType *buffer,
                                        services::Error &error);

    /** Get 2-class classification probabilities for a block of observations */
    inline void get2ClassProbabilities(size_t nFeatures, size_t startRow, size_t nRows, size_t nClasses,
                                       MicroTable *mtX, algorithmFPType *y,
                                       services::SharedPtr<classifier::prediction::Batch> simplePrediction,
                                       Model *model,
                                       algorithmFPType *rProb, services::Error &error);

    /** Compute matrix Q from the 2-class parobabilities */
    inline void computeQ(size_t nClasses, const algorithmFPType *rProb, algorithmFPType *Q);

    /** Calculate objective function of the Algorithm 2 from [1] */
    inline algorithmFPType computeObjFunc(size_t nClasses, algorithmFPType *p, algorithmFPType *rProb);

    /** Update multi-class probability estimates */
    inline void updateProbabilities(size_t nClasses, const algorithmFPType *Q,
                                    algorithmFPType *Qp, algorithmFPType *p);
};

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
