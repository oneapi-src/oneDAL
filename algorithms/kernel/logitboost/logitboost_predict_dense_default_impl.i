/* file: logitboost_predict_dense_default_impl.i */
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
//  Common functions for Logit Boost predictions calculation
//--
*/

#ifndef __LOGITBOOST_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __LOGITBOOST_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "logitboost_model.h"
#include "threading.h"
#include "daal_defines.h"

#include "service_memory.h"
#include "service_numeric_table.h"
#include "logitboost_impl.i"

using namespace daal::algorithms::logitboost::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
void LogitBoostPredictKernel<defaultDense, algorithmFPType, cpu>::compute( NumericTablePtr a,
        const Model *m, NumericTable *r, const Parameter *par )
{
    Parameter *parameter = const_cast<Parameter *>(par);
    size_t dim = a->getNumberOfColumns();       /* Number of features in input dataset */
    size_t n   = a->getNumberOfRows();          /* Number of observations in input dataset */
    size_t nc  = parameter->nClasses;           /* Number of classes */
    size_t M   = m->getIterations();            /* Number of terms of additive regression in the model */
    algorithmFPType *pred;
    algorithmFPType *F;      /* Additive function values */
    Model *boostModel = const_cast<Model *>(m);

    /* Allocate memory */
    pred = (algorithmFPType *) daal::services::daal_malloc (n * nc * sizeof(algorithmFPType));
    F    = (algorithmFPType *) daal::services::daal_malloc (n * nc * sizeof(algorithmFPType));
    if (!pred || !F)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed); return;
    }
    daal::services::internal::service_memset<algorithmFPType, cpu>(F, 0, n * nc);

    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;
    learnerPredict->inputBase->set(classifier::prediction::data, a);

    /* Calculate additive function values */
    for ( size_t m = 0; m < M; m++ )
    {
        for (size_t j = 0; j < nc; j++)
        {
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > predTable(
                new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(pred + j * n, 1, n));
            services::SharedPtr<classifier::prediction::Result> predictionRes(new classifier::prediction::Result());
            predictionRes->set(classifier::prediction::prediction, predTable);
            learnerPredict->setResult(predictionRes);
            services::SharedPtr<weak_learner::Model> learnerModel = boostModel->getWeakLearnerModel(m * nc + j);

            learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
            learnerPredict->computeNoThrow();
            if(learnerPredict->getErrors()->size() != 0) {this->_errors->add(learnerPredict->getErrors()->getErrors()); return;}
        }

        UpdateF<algorithmFPType, cpu>( dim, n, nc, pred, F );
    }

    /* Calculate classes labels for input data */
    int *cl;
    BlockDescriptor<int> block;
    r->getBlockOfColumnValues( 0, 0, n, writeOnly, block );
    cl = block.getBlockPtr();
    algorithmFPType fmax;

    for ( size_t i = 0; i < n; i++ )
    {
        int idx = 0;
        fmax = F[i * nc];
        for ( int j = 1; j < nc; j++ )
        {
            if ( F[i * nc + j] > fmax )
            {
                idx = j;
                fmax = F[i * nc + j];
            }
        }

        cl[i] = idx;
    }

    r->releaseBlockOfColumnValues( block );

    daal::services::daal_free (F);
    daal::services::daal_free (pred);
}

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
