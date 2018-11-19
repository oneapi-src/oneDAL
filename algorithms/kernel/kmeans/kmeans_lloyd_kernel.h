/* file: kmeans_lloyd_kernel.h */
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef _KMEANS_FPK_H
#define _KMEANS_FPK_H

#include "kmeans_types.h"
//#include "kmeans_batch.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansBatchKernel: public Kernel
{
public:
    services::Status compute(const NumericTable *const *a, const NumericTable *const *r, const Parameter *par);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansDistributedStep1Kernel: public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par);
    services::Status finalizeCompute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par);
};

template <Method method, typename algorithmFPType, CpuType cpu>
class KMeansDistributedStep2Kernel: public Kernel
{
public:
    services::Status compute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par);
    services::Status finalizeCompute(size_t na, const NumericTable *const *a, size_t nr, const NumericTable *const *r, const Parameter *par);
};

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal

#endif
