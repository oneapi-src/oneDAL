/* file: boosting_training_batch.h */
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
//  Implementation of base classes defining interface for training models
//  of the Boosting algorithms in batch mode.
//--
*/

#ifndef __BOOSTING_TRAINING_BATCH_H__
#define __BOOSTING_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
namespace boosting
{
/**
 * \brief Contains classes for training %boosting models.
 */
namespace training
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @defgroup boosting_training Training
 * \copydoc daal::algorithms::boosting::training
 * @ingroup boosting
 * @{
 */
/**
 * @defgroup boosting_training_batch Batch
 * @ingroup boosting_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__TRAINING__BATCH"></a>
 * \brief %Base class for training models of %boosting algorithms in the batch processing mode
 *
  * \par Enumerations
 *      - \ref classifier::training::InputId  Identifiers of input objects for %boosting algorithms
 *      - \ref classifier::training::ResultId Result identifiers for %boosting algorithm training
 *
 * \par References
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 *      - \ref classifier::training::interface1::Result "classifier::training::Result" class
 */
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef super::InputType                InputType;
    typedef algorithms::boosting::Parameter ParameterType;
    typedef super::ResultType               ResultType;

    virtual ~Batch() {}

    /**
     * Returns a pointer to the newly allocated %boosting classifier training algorithm with a copy of input objects
     * and parameters of this %boosting classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:
    virtual Batch * cloneImpl() const DAAL_C11_OVERRIDE = 0;
};
/** @} */
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace daal::algorithms::boosting::training
}
}
}
#endif // __BOOSTING_TRAINING_BATCH_H__
