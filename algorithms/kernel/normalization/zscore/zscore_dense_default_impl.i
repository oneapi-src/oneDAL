/* file: zscore_dense_default_impl.i */
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
//  Implementation of defaultDense method for zscore normalization algorithm
//--
*/

#ifndef __ZSCORE_DENSE_DEFAULT_IMPL_I__
#define __ZSCORE_DENSE_DEFAULT_IMPL_I__

#include "service_micro_table.h"
#include "service_math.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace internal
{

/* TLS structure with local arrays and variables */
template<typename algorithmFPType, CpuType cpu>
struct tls_data_t
{
    algorithmFPType* mean;
    algorithmFPType* variance;
    algorithmFPType  nvectors;
    int malloc_errors;

    tls_data_t(size_t nFeatures)
    {
        malloc_errors = 0;

        nvectors = 0;

        mean  = service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
        if(!mean) { malloc_errors++; }

        variance  = service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
        if(!variance) { malloc_errors++; }

    }

    ~tls_data_t()
    {
        if(mean)  { service_scalable_free<algorithmFPType,cpu>( mean );  mean = 0; }
        if(variance) { service_scalable_free<algorithmFPType,cpu>( variance ); variance = 0; }
    }
};


template<typename algorithmFPType, CpuType cpu>
int ZScoreKernel<algorithmFPType, defaultDense, cpu>::computeMeanVariance_thr( SharedPtr<NumericTable> inputTable,
                                                                              algorithmFPType* resultMean,
                                                                              algorithmFPType* resultVariance,
                                                                              daal::algorithms::Parameter *par
                                                                            )
{
    int errs = 0;

    size_t _nVectors  = inputTable->getNumberOfRows();
    size_t _nFeatures = inputTable->getNumberOfColumns();

    Parameter<algorithmFPType, defaultDense> *parameter = static_cast<Parameter<algorithmFPType, defaultDense> *>(par);

    parameter->moments->input.set(low_order_moments::data, inputTable);
    parameter->moments->parameter.estimatesToCompute = low_order_moments::estimatesMeanVariance;
    parameter->moments->computeNoThrow();
    if(parameter->moments->getErrors()->size() != 0)
    {
      errs++;
      this->_errors->add(ErrorMeanAndStandardDeviationComputing);
      return errs;
    }

    NumericTablePtr meanTable     = parameter->moments->getResult()->get(low_order_moments::mean);
    NumericTablePtr varianceTable = parameter->moments->getResult()->get(low_order_moments::variance);

    daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> meanBlock( meanTable.get(), 0, 1 );
    const algorithmFPType* meanArray = meanBlock.get();

    daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> varianceBlock( varianceTable.get(), 0, 1 );
    const algorithmFPType* varianceArray = varianceBlock.get();

    /* Convert array of variances to inverse sigma's */
   PRAGMA_IVDEP
   PRAGMA_VECTOR_ALWAYS
   PRAGMA_SIMD_ASSERT
    for(int j = 0; j < _nFeatures; j++)
    {
        resultMean[j]     = meanArray[j];
        resultVariance[j] = algorithmFPType(1.0) / daal::internal::Math<algorithmFPType, cpu>::sSqrt(varianceArray[j]);
    }

    return errs;
}

} // namespace daal::internal
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal

#endif
