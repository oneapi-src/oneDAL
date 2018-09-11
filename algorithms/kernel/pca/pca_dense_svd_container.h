/* file: pca_dense_svd_container.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_CONTAINER_H__
#define __PCA_DENSE_SVD_CONTAINER_H__

namespace daal
{
namespace algorithms
{
namespace pca
{
static inline internal::InputDataType getInputDataType(pca::Input *input)
{
    if(input == 0 || input->size() == 0)
    {
        return internal::nonNormalizedDataset;
    }

    data_management::NumericTable *a = static_cast<data_management::NumericTable *>(input->get(data).get());
    if(input->isCorrelation())
    {
        return internal::correlation;
    }
    else if(a->isNormalized(data_management::NumericTableIface::standardScoreNormalized))
    {
        return internal::normalizedDataset;
    }
    else
    {
        return internal::nonNormalizedDataset;
    }
}

}
}
} // namespace daal
#endif
