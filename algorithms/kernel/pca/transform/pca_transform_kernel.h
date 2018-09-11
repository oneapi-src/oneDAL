/* file: pca_transform_kernel.h */
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
//  Declaration of template function that computes pca transformation
//--
*/

#ifndef __PCA_TRANSFORM_KERNEL_H__
#define __PCA_TRANSFORM_KERNEL_H__

#include "pca_transform_batch.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_blas.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace internal
{

template <typename algorithmFPType, transform::Method method, CpuType cpu>
class TransformKernel : public Kernel
{
public:
    /**
     *  \brief Compute PCA transformation.
     *
     *  \param data[in]             Matrix of input vectors X
     *  \param eigenvectors[in]     PCA eigenvectors
     *  \param means[in]            PCA means
     *  \param variances[in]        PCA variances
     *  \param eigenvalues[in]        PCA eigenvalues
     *  \param transformedData[out] Transformed data
     */
    services::Status compute(NumericTable& data,
                             NumericTable& eigenvectors,
                             NumericTable* pMeans,
                             NumericTable* pVariances,
                             NumericTable* pEigenvalues,
                             NumericTable& transformedData);


    /**
    *  \brief Function that computes PCA transformation
    *         for a block of input data rows
    *
    *  \param numRows[in]          Number of input data rows
    *  \param numFeatures[in]      Number of features in input data row
    *  \param numComponents[in]    Number of components
    *  \param dataBlock[in]        Block of input data rows
    *  \param eigenvectors[in]     Eigenvectors
    *  \param resultBlock[out]     Resulting block of responses
    */
    void computeTransformedBlock(DAAL_INT *numRows, DAAL_INT *numFeatures, DAAL_INT *numComponents,
                                 const algorithmFPType *dataBlock,
                                 const algorithmFPType *eigenvectors,
                                 algorithmFPType *resultBlock);

    static const size_t _numRowsInBlock = 256;

};
} // namespace internal
} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
