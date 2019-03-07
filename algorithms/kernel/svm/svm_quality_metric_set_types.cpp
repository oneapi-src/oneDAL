/* file: svm_quality_metric_set_types.cpp */
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
//  Interface for the svm algorithm quality metrics
//--
*/

#include "svm_quality_metric_set_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace quality_metric_set
{
namespace interface1
{

/**
 * Returns the result of the quality metrics algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
classifier::quality_metric::binary_confusion_matrix::ResultPtr ResultCollection::getResult(QualityMetricId id) const
{
    return staticPointerCast<classifier::quality_metric::binary_confusion_matrix::Result, SerializationIface>((*this)[(size_t)id]);
}

/**
 * Returns the input object of the quality metrics algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
classifier::quality_metric::binary_confusion_matrix::InputPtr InputDataCollection::getInput(QualityMetricId id) const
{
    return staticPointerCast<classifier::quality_metric::binary_confusion_matrix::Input, algorithms::Input>(
            algorithms::quality_metric_set::InputDataCollection::getInput((size_t)id));
}


} //namespace interface1
} //namespace quality_metric_set
} //namespace svm
} //namespace algorithms
} //namespace daal
