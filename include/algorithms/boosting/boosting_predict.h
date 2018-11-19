/* file: boosting_predict.h */
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
//  Implementation of base classes defining interface for prediction
//  based on Boosting algorithm model.
//--
*/

#ifndef __BOOSTING_PREDICT_H__
#define __BOOSTING_PREDICT_H__

#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/boosting/boosting_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup boosting Boosting
 * \brief Contains classes to work with boosting algorithms
 * @ingroup classification
 */
namespace boosting
{
/**
 * \brief Contains classes for prediction based on %boosting models.
 */
namespace prediction
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @defgroup boosting_prediction Prediction
 * \copydoc daal::algorithms::boosting::prediction
 * @ingroup boosting
 * @{
 */
/**
 * @defgroup boosting_prediction_batch Batch
 * @ingroup boosting_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__PREDICTION__BATCH"></a>
 * \brief %Base class for predicting results of %boosting classifiers
 *
 * \par Enumerations
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input NumericTable objects for %boosting algorithms
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects for %boosting algorithms
 *      - \ref classifier::prediction::ResultId             Result identifiers for %boosting algorithm training
 *
 * \par References
 *      - \ref classifier::prediction::interface1::Input "classifier::prediction::Input" class
 *      - \ref classifier::prediction::interface1::Result "classifier::prediction::Result" class
 */
class Batch : public classifier::prediction::Batch
{
public:
    typedef classifier::prediction::Batch super;

    typedef super::InputType                InputType;
    typedef algorithms::boosting::Parameter ParameterType;
    typedef super::ResultType               ResultType;

    Batch() {}

    /**
     * Constructs %boosting classifier prediction algorithm by copying input objects and parameters
     * of another %boosting classifier prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other) : classifier::prediction::Batch(other) {}

    virtual ~Batch() {}

    /**
     * Returns a pointer to the newly allocated %boosting classifier prediction algorithm with a copy of input objects
     * and parameters of this %boosting classifier prediction algorithm
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

} // namespace daal::algorithms::boosting::prediction
}
}
} // namespace daal
#endif // __BOOSTING_PREDICT_H__
