/* file: stump_training_types.h */
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
//  Implementation of the interface of the decision stump training algorithm.
//--
*/

#ifndef __STUMP_TRAINING_TYPES_H__
#define __STUMP_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/weak_learner/weak_learner_training_types.h"
#include "algorithms/stump/stump_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to work with the decision stump training algorithm
 */
namespace stump
{
/**
 * @defgroup stump_training Training
 * \copydoc daal::algorithms::stump::training
 * @ingroup stump
 * @{
 */
/**
 * \brief Contains classes to train the decision stump model
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__TRAINING__METHOD"></a>
 * Available methods to train the decision stump model
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the decision stump training algorithm
 * in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::weak_learner::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    virtual ~Result() {}

    /**
     * Returns the model trained with the Stump algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the Stump algorithm
     */
    daal::algorithms::stump::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Sets the result of the training stage of the stump algorithm
     * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
     * \param[in] value   Pointer to the training result
     */
    void set(classifier::training::ResultId id, daal::algorithms::stump::ModelPtr &value);

    /**
     * Allocates memory to store final results of the decision stump training algorithm
     * \tparam algorithmFPType  Data type to store prediction results
     * \param[in] input         %Input objects for the decision stump training algorithm
     * \param[in] parameter     Parameters of the decision stump training algorithm
     * \param[in] method        Decision stump training method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Check the correctness of the Result object
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameters structure
     * \param[in] method    Algorithm computation method
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace daal::algorithms::stump::training
/** @} */
}
}
} // namespace daal
#endif // __STUMP_TRAINING_TYPES_H__
