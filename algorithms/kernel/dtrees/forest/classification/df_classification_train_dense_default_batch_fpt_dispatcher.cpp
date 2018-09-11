/* file: df_classification_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of decision forest classification container.
//--
*/

#include "df_classification_train_container.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(decision_forest::classification::training::BatchContainer, batch, DAAL_FPTYPE, \
    decision_forest::classification::training::defaultDense)

}
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace interface1
{
template<>
DAAL_EXPORT services::Status Batch<DAAL_FPTYPE, decision_forest::classification::training::defaultDense>::checkComputeParams()
{
    services::Status s = classifier::training::Batch::checkComputeParams();
    if(!s)
        return s;
    const auto x = input.get(classifier::training::data);
    const auto nFeatures = x->getNumberOfColumns();
    DAAL_CHECK_EX(parameter.featuresPerNode <= nFeatures,
        services::ErrorIncorrectParameter, services::ParameterName, featuresPerNodeStr());
    const size_t nSamplesPerTree(parameter.observationsPerTreeFraction*x->getNumberOfRows());
    DAAL_CHECK_EX(nSamplesPerTree > 0,
        services::ErrorIncorrectParameter, services::ParameterName, observationsPerTreeFractionStr());
    return s;
}
}
}
}
}
}
} // namespace daal
