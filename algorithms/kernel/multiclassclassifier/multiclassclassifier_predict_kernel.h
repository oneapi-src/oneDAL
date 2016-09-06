/* file: multiclassclassifier_predict_kernel.h */
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
//  Declaration of template function that computes prediction results using
//  Multi-class classifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_FPK_H__
#define __MULTICLASSCLASSIFIER_PREDICT_FPK_H__

#include "numeric_table.h"
#include "model.h"
#include "algorithm.h"
#include "multi_class_classifier_types.h"
#include "service_defines.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{

template<prediction::Method pmethod, training::Method tmethod, typename AlgorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel : public Kernel
{
    void compute(const NumericTable *a, const daal::algorithms::Model *m, NumericTable *r,
                 const daal::algorithms::Parameter *par);
};


} // namespace internal

} // namespace prediction

} // namespace multi_class_classifier

} // namespace algorithms

} // namespace daal


#endif
