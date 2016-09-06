/* file: outlierdetection_univariate_dense_default_impl.i */
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
//  Implementation of univariate outlier detection
//--
*/

#ifndef __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_IMPL_I__
#define __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_IMPL_I__

#include "numeric_table.h"
#include "outlier_detection_univariate_types.h"

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"

#include "outlierdetection_univariate_dense_default_kernel.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
void OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu>::
    compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par)
{

    /* Create micro-tables for input data and output results */
    BlockMicroTable<AlgorithmFPType, readOnly, cpu> mtA(a);
    BlockMicroTable<AlgorithmFPType, writeOnly, cpu> mtR(r);
    size_t nFeatures = mtA.getFullNumberOfColumns();
    size_t nVectors = mtA.getFullNumberOfRows();

    /* Check algorithm's parameters */
    bool insideAllocatedParameter = false;
    Parameter *innerPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    if (innerPar->initializationProcedure.get() == NULL) // TODO: remove later
    {
        insideAllocatedParameter = true;
        innerPar = new Parameter();
        innerPar->initializationProcedure = services::SharedPtr<univariate_outlier_detection::InitIface>(new TemporaryInitialization<cpu>
                                                                                                         (nFeatures));
    }

    /* Get algorithm's parameters */
    Parameter *odPar = static_cast<Parameter *>(innerPar);
    InitIface *initProcedure = odPar->initializationProcedure.get();

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > locationTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, 1));

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > scatterTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, 1));

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > thresholdTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, 1));

    (*initProcedure)(const_cast<NumericTable *>(a), locationTable.get(), scatterTable.get(), thresholdTable.get());

    /* Allocate memory for storing intermediate results */
    AlgorithmFPType *invScatter = (AlgorithmFPType *)daal::services::daal_malloc(nFeatures * sizeof(AlgorithmFPType));
    if (invScatter == NULL) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Calculate results */
    computeInternal(nFeatures, nVectors, mtA, mtR,
                    locationTable->getArray(),
                    scatterTable->getArray(),
                    invScatter,
                    thresholdTable->getArray());

    /* Release memory */
    daal::services::daal_free(invScatter);
    if(insideAllocatedParameter) { delete innerPar; }
}

} // namespace internal

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
