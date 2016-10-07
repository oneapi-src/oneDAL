/* file: lrn_layer.cpp */
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
//  Implementation of lrn calculation algorithm and types methods.
//--
*/

#include "lrn_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace interface1
{
/**
*  Constructs parameters of the local response normalization layer
*  \param[in] dimension_ Numeric table of size 1 x 1 with index of type size_t to calculate local response normalization.
*  \param[in] kappa_     Value of hyper-parameter kappa
*  \param[in] alpha_     Value of hyper-parameter alpha
*  \param[in] beta_      Value of hyper-parameter beta
*  \param[in] nAdjust_   Value of hyper-parameter n
*/
Parameter::Parameter(data_management::NumericTablePtr dimension_,
                     const double kappa_,
                     const double alpha_,
                     const double beta_ ,
                     const size_t nAdjust_) :
    dimension(dimension_),
    kappa(kappa_),
    alpha(alpha_),
    beta(beta_),
    nAdjust(nAdjust_)
{};

    /**
 * Checks the correctness of the parameter
 */
void Parameter::check() const
{
    services::SharedPtr<services::Error> error(new services::Error());
    if(dimension.get() == NULL)
    {
        error->setId(services::ErrorIncorrectParameter);
    }
    else if(dimension->getNumberOfRows() != 1)
    {
        error->setId(services::ErrorIncorrectNumberOfObservations);
    }
    else if(dimension->getNumberOfColumns() != 1)
    {
        error->setId(services::ErrorIncorrectNumberOfFeatures);
    }
    if(error->id() != services::NoErrorMessageFound)
    {
        error->addStringDetail(services::ArgumentName, "dimension");
        this->_errors->add(error);
    }
}

}// namespace interface1
}// namespace lrn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
