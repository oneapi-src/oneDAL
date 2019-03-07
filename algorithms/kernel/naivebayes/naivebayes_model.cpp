/* file: naivebayes_model.cpp */
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
//  Implementation of multinomial naive bayes algorithm parameters.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace interface1
{

PartialModel::PartialModel() : _nObservations(0) { }

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());

    if(priorClassEstimates)
    {
        s |= checkNumericTable(priorClassEstimates.get(), priorClassEstimatesStr(), 0, 0, 1, nClasses);
    }

    return s;
}

} // namespace interface1
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
