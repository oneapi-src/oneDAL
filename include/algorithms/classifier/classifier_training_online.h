/* file: classifier_training_online.h */
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

#ifndef __CLASSIFIER_TRAINING_ONLINE_H__
#define __CLASSIFIER_TRAINING_ONLINE_H__

#include "services/daal_defines.h"

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
 * @defgroup classifier_training_online Online
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__ONLINE"></a>
 *  \brief Algorithm class for training the classifier model in the online processing mode
 *
 * \par Enumerations
 *      - \ref InputId  %Input objects of the classifier model training algorithm
 *      - \ref ResultId Results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class DAAL_EXPORT Online : public Training<online>
{
public:
    typedef algorithms::classifier::training::Input         InputType;
    typedef algorithms::classifier::Parameter               ParameterType;
    typedef algorithms::classifier::training::Result        ResultType;
    typedef algorithms::classifier::training::PartialResult PartialResultType;

    InputType input;     /*!< %Input objects of the algorithm */

    Online()
    {
        initialize();
    }

    /**
     * Constructs a classifier training algorithm by copying input objects and parameters
     * of another classifier training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online &other) : input(other.input)
    {
        initialize();
    }

    virtual ~Online() {}

    /**
     * Registers user-allocated memory for storing partial training results
     * \param[in] partialResult Structure for storing partial results
     * \param[in] initFlag Flag if partial result initialized or not
     */
    services::Status setPartialResult(const PartialResultPtr &partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

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
     * Returns the structure that contains computed partial results of the classification algorithm
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Returns the structure that contains results of the classification algorithm
     * \return Structure that contains computed results
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online> clone() const
    {
        return services::SharedPtr<Online>(cloneImpl());
    }

protected:
    PartialResultPtr _partialResult;
    ResultPtr _result;

    void initialize()
    {
        _in = &input;
    }
    virtual Online * cloneImpl() const DAAL_C11_OVERRIDE = 0;
};
/** @} */
} // namespace interface1
using interface1::Online;

}
}
}
}
#endif
