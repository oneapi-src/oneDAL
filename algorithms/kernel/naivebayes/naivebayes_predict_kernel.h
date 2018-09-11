/* file: naivebayes_predict_kernel.h */
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

#ifndef _NAIVEBAYES_ASSIGN_FPK_H
#define _NAIVEBAYES_ASSIGN_FPK_H

#include "multinomial_naive_bayes_model.h"
#include "multinomial_naive_bayes_predict_types.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
namespace internal
{

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesPredictKernel : public Kernel
{
public:
    services::Status compute(const NumericTable *a, const Model *m, NumericTable *r, const Parameter *par);
};

} // namespace internal
} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal

#endif
