/* file: weak_learner_training_types.h */
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
//  Implementation of the base classes used on the training stage
//  of weak learners algorithms
//--
*/

#ifndef __WEAK_LEARNER_TRAINING_TYPES_H__
#define __WEAK_LEARNER_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/weak_learner/weak_learner_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for working with weak learners
 */
namespace weak_learner
{
/**
 * @defgroup weak_learner_training Training
 * \copydoc daal::algorithms::weak_learner::training
 * @ingroup weak_learner
 * @{
 */
/**
 * \brief Contains classes for training models of the weak learners algorithms
 */
namespace training
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with compute() method of Batch
 *        or finalizeCompute() method of Online and Distributed weak learners algorithms
 */
class DAAL_EXPORT Result : public daal::algorithms::classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the weak learner  algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the weak learner  algorithm
     */
    daal::algorithms::weak_learner::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Sets the result of the training stage of the weak learner algorithm
     * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
     * \param[in] value   Pointer to the training result
     */
    void set(classifier::training::ResultId id, daal::algorithms::weak_learner::ModelPtr &value);

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace daal::algorithms::weak_learner::training
/** @} */
} // namespace daal::algorithms::weak_learner
} // namespace daal::algorithms
} // namespace daal
#endif
