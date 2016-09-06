/* file: outlierdetection_multivariate_dense_default_impl.i */
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
//  Implementation of multivariate outlier detection
//--
*/

#ifndef __MULTIVARIATE_OUTLIER_DETECTION_DENSE_DEFAULT_IMPL_I__
#define __MULTIVARIATE_OUTLIER_DETECTION_DENSE_DEFAULT_IMPL_I__

#include "numeric_table.h"
#include "outlier_detection_multivariate_types.h"

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_blas.h"
#include "service_lapack.h"

#include "outlierdetection_multivariate_dense_default_kernel.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multivariate_outlier_detection
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
inline void OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu>::
    mahalanobisDistance(size_t nFeatures, size_t nVectors, AlgorithmFPType *data,
                        AlgorithmFPType *location, AlgorithmFPType *invScatter, AlgorithmFPType *distance,
                        AlgorithmFPType *buffer)
{
    AlgorithmFPType *dataCen            = buffer;
    AlgorithmFPType *dataCenInvScatter  = buffer + nFeatures * nVectors;

    MKL_INT dim = (MKL_INT)nFeatures;
    MKL_INT n   = (MKL_INT)nVectors;
    char side = 'L';
    char uplo = 'U';
    AlgorithmFPType one  = (AlgorithmFPType)1.0;
    AlgorithmFPType zero = (AlgorithmFPType)0.0;
    MKL_INT info;

    AlgorithmFPType *dataPtr    = data;
    AlgorithmFPType *dataCenPtr = dataCen;
    for (size_t i = 0; i < nVectors; i++, dataPtr += nFeatures, dataCenPtr += nFeatures)
    {
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            dataCenPtr[j] = dataPtr[j] - location[j];
        }
    }

    Blas<AlgorithmFPType, cpu>::xsymm(&side, &uplo, &dim, &n, &one, invScatter, &dim, dataCen, &dim, &zero,
                       dataCenInvScatter, &dim);

    dataCenPtr = dataCen;
    AlgorithmFPType *dataCenInvScatterPtr = dataCenInvScatter;
    for (size_t i = 0; i < nVectors; i++, dataCenPtr += nFeatures, dataCenInvScatterPtr += nFeatures)
    {
        distance[i] = zero;
      PRAGMA_IVDEP
      PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            distance[i] += dataCenPtr[j] * dataCenInvScatterPtr[j];
        }
    }
}

template <typename AlgorithmFPType, CpuType cpu>
inline void OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu>::
    computeInternal(size_t nFeatures, size_t nVectors,
                    BlockMicroTable  <AlgorithmFPType, readOnly,  cpu> &mtA,
                    FeatureMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                    AlgorithmFPType *location, AlgorithmFPType *scatter, AlgorithmFPType threshold,
                    AlgorithmFPType *buffer)
{
    AlgorithmFPType one  = (AlgorithmFPType)1.0;
    AlgorithmFPType zero = (AlgorithmFPType)0.0;
    AlgorithmFPType *invScatter         = buffer;

    for (size_t i = 0; i < nFeatures * nFeatures; i++)
    {
        invScatter[i] = scatter[i];
    }

    /* Calculate inverse of data variance-covariance matrix */
    MKL_INT dim = (MKL_INT)nFeatures;
    char uplo = 'U';
    MKL_INT info;
    Lapack<AlgorithmFPType, cpu>::xpotrf(&uplo, &dim, invScatter, &dim, &info);
    if (info != 0) { this->_errors->add(services::ErrorOutlierDetectionInternal); return; }

    Lapack<AlgorithmFPType, cpu>::xpotri(&uplo, &dim, invScatter, &dim, &info);
    if (info != 0) { this->_errors->add(services::ErrorOutlierDetectionInternal); return; }

    size_t nBlocks = nVectors / blockSize;
    if (nBlocks * blockSize < nVectors)
    {
        nBlocks++;
    }

    /* Process input data table in blocks */
    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        size_t startRow = iBlock * blockSize;
        size_t nRowsInBlock = blockSize;
        if (startRow + nRowsInBlock > nVectors)
        {
            nRowsInBlock = nVectors - startRow;
        }
        MKL_INT n = (MKL_INT)nRowsInBlock;

        AlgorithmFPType *data, *weight;
        mtA.getBlockOfRows(startRow, nRowsInBlock, &data);
        mtR.getBlockOfColumnValues(0, startRow, nRowsInBlock, &weight);

        /* Calculate mahalanobis distances for a block of observations */
        mahalanobisDistance(nFeatures, nRowsInBlock, data, location, invScatter, weight,
                            buffer + nFeatures * nFeatures);

        for (size_t i = 0; i < nRowsInBlock; i++)
        {
            if (daal::internal::Math<AlgorithmFPType,cpu>::sSqrt(weight[i]) > threshold)
            {
                weight[i] = zero;
            }
            else
            {
                weight[i] = one;
            }
        }

        mtA.release();
        mtR.release();
    }
}

template <typename AlgorithmFPType, CpuType cpu>
void OutlierDetectionKernel<AlgorithmFPType, defaultDense, cpu>::
    compute(const NumericTable *a, NumericTable *r, const daal::algorithms::Parameter *par)
{

    /* Create micro-tables for input data and output results */
    BlockMicroTable  <AlgorithmFPType, readOnly,  cpu> mtA(a);
    FeatureMicroTable<AlgorithmFPType, writeOnly, cpu> mtR(r);
    size_t nFeatures = mtA.getFullNumberOfColumns();
    size_t nVectors  = mtA.getFullNumberOfRows();

    /* Check algorithm's parameters */
    bool insideAllocatedParameter = false;
    Parameter<defaultDense> *innerPar = static_cast<Parameter<defaultDense> *>(const_cast<daal::algorithms::Parameter *>
                                                                               (par));
    if (innerPar->initializationProcedure.get() == NULL) // TODO: remove later
    {
        insideAllocatedParameter = true;
        innerPar = new Parameter<defaultDense>();
        innerPar->initializationProcedure = services::SharedPtr<multivariate_outlier_detection::InitIface>
                                            (new TemporaryInitialization<cpu>(nFeatures));
    }

    InitIface *initProcedure = innerPar->initializationProcedure.get();

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > locationTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, 1));

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > scatterTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(nFeatures, nFeatures));

    services::SharedPtr<daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu> > thresholdTable(
        new daal::internal::HomogenNumericTableCPU<AlgorithmFPType, cpu>(1, 1));

    (*initProcedure)(const_cast<NumericTable *>(a), locationTable.get(), scatterTable.get(), thresholdTable.get());

    AlgorithmFPType threshold = (thresholdTable->getArray())[0];

    /* Allocate memory for storing intermediate results */
    size_t bufferSize = nFeatures * nFeatures + 2 * nFeatures * nVectors;
    AlgorithmFPType *buffer = (AlgorithmFPType *)daal::services::daal_malloc(bufferSize * sizeof(AlgorithmFPType));
    if (buffer == NULL) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Calculate results */
    computeInternal(nFeatures, nVectors, mtA, mtR,
                    locationTable->getArray(),
                    scatterTable->getArray(),
                    threshold,
                    buffer);

    /* Release memory */
    daal::services::daal_free(buffer);
    if(insideAllocatedParameter) { delete innerPar; }
}

} // namespace internal

} // namespace multivariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
