/* file: implicit_als_train_dense_default_batch_common.i */
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
//  Implementation of common computational kernels of impicit ALS training algorithm
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_COMMON_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_COMMON_I__

#include "service_blas.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
void computeXtX(
    size_t *nRows, size_t *nCols, algorithmFPType *beta, algorithmFPType *x, size_t *ldx,
    algorithmFPType *xtx, size_t *ldxtx)
{
    /* SYRK parameters */
    char uplo = 'U';
    char trans = 'N';
    algorithmFPType alpha = 1.0;

    daal::internal::Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)nCols, (DAAL_INT *)nRows, &alpha, x, (DAAL_INT *)ldx, beta,
                       xtx, (DAAL_INT *)ldxtx);
}

}
}
}
}
}

#endif
