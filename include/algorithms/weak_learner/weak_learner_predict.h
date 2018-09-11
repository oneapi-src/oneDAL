/* file: weak_learner_predict.h */
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
//  Implementation of base classes defining the interface for prediction
//  based on the weak learner model.
//--
*/

#ifndef __WEAK_LEARNER_PREDICT_H__
#define __WEAK_LEARNER_PREDICT_H__

#include "algorithms/classifier/classifier_predict.h"

namespace daal
{
namespace algorithms
{
namespace weak_learner
{
/**
 * @defgroup weak_learner_prediction Prediction
 * \copydoc daal::algorithms::weak_learner::prediction
 * @ingroup weak_learner
 * @{
 */
/**
 * \brief Contains classes to make predictions based on the weak learner model
 */
namespace prediction
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @defgroup weak_learner_prediction_batch Batch
 * @ingroup weak_learner_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__PREDICTION__BATCH"></a>
 * \brief %Base class for making predictions based on the weak learner model
 *
 * \par Enumerations
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects
 *      - \ref classifier::prediction::ResultId             Identifiers of prediction results
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Result "Result" class
 */
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef super::InputType     InputType;
    typedef super::ParameterType ParameterType;
    typedef super::ResultType    ResultType;

    Batch() {}

    /**
     * Constructs algorithm for making predictions based on the weak learner model
     * by copying input objects and parameters of another algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other) : classifier::prediction::Batch(other) {}

    virtual ~Batch() {}

    /**
     * Returns a pointer to the newly allocated algorithm for making predictions based on the weak learner model
     * with a copy of input objects and parameters of this algorithm
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
} // namespace interface1
using interface1::Batch;

} // namespace daal::algorithms::weak_learner::prediction
}
}
} // namespace daal
#endif // __WEAK_LEARNER_PREDICT_H__
