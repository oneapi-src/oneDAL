/* file: classifier_predict.h */
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
//  Implementation of the prediction stage of the classification algorithm interface.
//--
*/

#ifndef __CLASSIFIER_PREDICT_H__
#define __CLASSIFIER_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{

namespace interface1
{
/**
 * @defgroup classifier_prediction_batch Batch
 * @ingroup prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__BATCH"></a>
 *  \brief Base class for making predictions based on the model of the classification algorithms
 *
 *  \par Enumerations
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input NumericTable objects
 *                                                          of the classifier prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects
 *                                                          of the classifier prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of prediction results of the classifier algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class Batch : public daal::algorithms::Prediction
{
public:
    typedef algorithms::classifier::prediction::Input  InputType;
    typedef algorithms::classifier::Parameter          ParameterType;
    typedef algorithms::classifier::prediction::Result ResultType;

    Batch()
    {
        initialize();
    }

    /**
     * Constructs a classifier prediction algorithm by copying input objects and parameters
     * of another classifier prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch &other)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the classifier prediction algorithm
     * \return %Input objects for the classifier prediction algorithm
     */
    virtual InputType * getInput() = 0;

    /**
     * Returns the structure that contains computed prediction results
     * \return Structure that contains computed prediction results
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory for storing the prediction results
     * \param[in] result Structure for storing the prediction results
     *
     * \return Status of computation
     */
    services::Status setResult(const ResultPtr &result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated classifier prediction algorithm with a copy of input objects
     * and parameters of this classifier prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:

    void initialize()
    {
        _result.reset(new ResultType());
    }
    virtual Batch * cloneImpl() const DAAL_C11_OVERRIDE = 0;
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::Batch;

}
}
}
}
#endif
