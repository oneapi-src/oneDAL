/* file: weak_learner_model.h */
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
//  Implementation of the class defining the weak learner model.
//--
*/

#ifndef __WEAK_LEARNER_MODEL_H__
#define __WEAK_LEARNER_MODEL_H__

#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup weak_learner Weak Learner
 * \copydoc daal::algorithms::weak_learner
 * @ingroup boosting
 * @{
 */
namespace weak_learner
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__PARAMETER"></a>
 * \brief %Base class for the input objects of the weak learner training and prediction algorithm
 *
 * \snippet weak_learner/weak_learner_model.h Parameter source code
 */
/* [Parameter source code] */
class Parameter : public classifier::Parameter
{
public:
    Parameter() {}
    virtual ~Parameter() {}
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__MODEL"></a>
 * \brief %Base class for the weak learner model
 */
class Model : public classifier::Model
{
public:
    Model() {}
    virtual ~Model() {}
};
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace daal::algorithms::weak_learner
/** @} */
}
} // namespace daal
#endif // __WEAK_LEARNER_MODEL_H__
