/* file: dropout_layer.cpp */
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
//  Implementation of dropout calculation algorithm and types methods.
//--
*/

#include "dropout_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace dropout
{
namespace interface1
{
/**
*  Constructs parameters of the dropout layer
*  \param[in] retainRatio_ Probability that any particular element is retained
*  \param[in] seed_        Seed for mask elements random generation
*/
Parameter::Parameter(const double retainRatio_, const size_t seed_) : retainRatio(retainRatio_), seed(seed_)
{};

/**
 * Checks the correctness of the parameter
 */
void Parameter::check() const
{
    if(retainRatio <= 0.0)
    {
        services::SharedPtr<services::Error> error(new services::Error());
        error->setId(services::ErrorIncorrectParameter);
        error->addStringDetail(services::ArgumentName, "retainRatio");
        this->_errors->add(error);
    }
}

}// namespace interface1
}// namespace dropout
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
