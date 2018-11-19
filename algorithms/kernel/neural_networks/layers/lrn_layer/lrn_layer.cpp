/* file: lrn_layer.cpp */
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
services::Status Parameter::check() const
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
        return services::Status(error);
    }
    return services::Status();
}

}// namespace interface1
}// namespace lrn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
