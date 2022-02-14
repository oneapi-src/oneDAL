/* file: multiclassclassifier_predict_mccwu_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Instantiation of Multi-class classifier prediction algorithm container.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_predict.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_predict_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(multi_class_classifier::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      multi_class_classifier::prediction::multiClassClassifierWu, multi_class_classifier::training::oneAgainstOne)

} // namespace algorithms
} // namespace daal
