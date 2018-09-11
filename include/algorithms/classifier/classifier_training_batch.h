/* file: classifier_training_batch.h */
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
//  Implementation of the interface for the classifier model training algorithm.
//--
*/

#ifndef __CLASSIFIER_TRAINING_BATCH_H__
#define __CLASSIFIER_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{

namespace interface1
{
/**
 * @defgroup classifier_training_batch Batch
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__BATCH"></a>
 * \brief Algorithm class for training the classifier model
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects of the classifier model training algorithm
 *      - \ref ResultId Identifiers of results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class Batch : public Training<batch>
{
public:
    typedef algorithms::classifier::training::Input  InputType;
    typedef algorithms::classifier::Parameter        ParameterType;
    typedef algorithms::classifier::training::Result ResultType;

    virtual ~Batch()
    {}

    /**
     * Get input objects for the classifier model training algorithm
     * \return %Input objects for the classifier model training algorithm
     */
    virtual InputType * getInput() = 0;

    /**
     * Registers user-allocated memory for storing results of the classifier model training algorithm
     * \param[in] res    Structure for storing results of the classifier model training algorithm
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the trained classifier model
     * \return Structure that contains the trained classifier model
     */
    ResultPtr getResult() { return _result; }

    /**
     * Resets the results of the classifier model training algorithm
     * \return Status of the operation
     */
    virtual services::Status resetResult() = 0;

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const
    {
        return services::SharedPtr<Batch>(cloneImpl());
    }

protected:

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
