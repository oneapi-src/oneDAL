/* file: outlierdetection_univariate_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of univariate outlier detection
//--
*/

#ifndef __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_IMPL_I__
#define __UNIVAR_OUTLIERDETECTION_DENSE_DEFAULT_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace univariate_outlier_detection
{
namespace internal
{

template <typename algorithmFPType, Method method, CpuType cpu>
Status OutlierDetectionKernel<algorithmFPType, method, cpu>::
compute(NumericTable &dataTable, NumericTable &resultTable,
        NumericTable *locationTable,
        NumericTable *scatterTable,
        NumericTable *thresholdTable)
{
    /* Create micro-tables for input data and output results */
    size_t nFeatures = dataTable.getNumberOfColumns();
    size_t nVectors = resultTable.getNumberOfRows();

    TArray<algorithmFPType, cpu> locationPtr, scatterPtr, thresholdPtr;
    ReadRows<algorithmFPType, cpu> locationBlock(locationTable), scatterBlock(scatterTable), thresholdBlock(thresholdTable);

    algorithmFPType* locationArray  = (locationTable) ?  const_cast<algorithmFPType *>(locationBlock.next(0, 1)):  locationPtr.reset(nFeatures);
    algorithmFPType* scatterArray   = (scatterTable) ?   const_cast<algorithmFPType *>(scatterBlock.next(0, 1)):   scatterPtr.reset(nFeatures);
    algorithmFPType* thresholdArray = (thresholdTable) ? const_cast<algorithmFPType *>(thresholdBlock.next(0, 1)): thresholdPtr.reset(nFeatures);

    DAAL_CHECK(locationArray && scatterArray && thresholdArray, ErrorMemoryAllocationFailed)

    if(!locationTable || !scatterTable || !thresholdTable)
    {
        defaultInitialization(locationArray, scatterArray, thresholdArray, nFeatures);
    }

    /* Allocate memory for storing intermediate results */
    TArray<algorithmFPType, cpu> invScatterPtr(nFeatures);
    DAAL_CHECK(invScatterPtr.get(), ErrorMemoryAllocationFailed)

    /* Calculate results */
    return computeInternal(nFeatures, nVectors, dataTable, resultTable,
                           locationArray,
                           scatterArray,
                           invScatterPtr.get(),
                           thresholdArray);
}

template <typename algorithmFPType, Method method, CpuType cpu>
void OutlierDetectionKernel<algorithmFPType, method, cpu>::defaultInitialization(
    algorithmFPType *locationArray,
    algorithmFPType *scatterArray,
    algorithmFPType *thresholdArray,
    const size_t nFeatures)
{
    for(size_t i = 0; i < nFeatures; i++)
    {
        locationArray[i]  = 0.0;
        scatterArray[i]   = 1.0;
        thresholdArray[i] = 3.0;
    }
}

} // namespace internal

} // namespace univariate_outlier_detection

} // namespace algorithms

} // namespace daal

#endif
