/* file: logistic_regression_model_builder.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the class defining the logistic regression model builder
//--
*/

#ifndef __LOGISTIC_REGRESSION_MODEL_BUILDER_H__
#define __LOGISTIC_REGRESSION_MODEL_BUILDER_H__

#include "algorithms/logistic_regression/logistic_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup logistic_regression Logistic regression
 * \copydoc daal::algorithms::logistic_regression
 * @ingroup classification
 */
/**
 * \brief Contains classes for the logistic regression algorithm
 */
namespace logistic_regression
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup logistic_regression
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL__BUILDER"></a>
 * \brief %Class for building model of the logistic regression algorithm
 *
 * \tparam modelFPType  Data type to store logistic regression model data, double or float
 *
 */
template<typename modelFPType = DAAL_ALGORITHM_FP_TYPE>
class DAAL_EXPORT ModelBuilder
{
public:

    /**
     * Constructs the Logistic Regression model builder
     * \param[in] nFeatures      Number of features in training data
     * \param[in] nClasses       Number of classes in training data
     */
    ModelBuilder(size_t nFeatures, size_t nClasses);

    /**
     *  Method to set betas to model via random access iterator, last - first value have to be equal to (_nFeatures)*_nClasses
     *  in case when intercept flag is suppose to be false and (_nFeatures + 1)*_nClasses when intercept flag is true
     * \tparam RandomIterator Random access iterator type for access to values of suport vectors
     *  \param[in] first      Iterator which point to first element of support vectors
     *  \param[in] last       Iterator which point to last element of support vectors
     */
    template<typename RandomIterator>
    void setBeta(RandomIterator first, RandomIterator last)
    {
        data_management::BlockDescriptor<modelFPType> pBlock;
        const size_t nVectorsBeta = _nClasses == 2 ? 1 : _nClasses;
        _modelPtr->getBeta()->getBlockOfRows(0, nVectorsBeta, data_management::readWrite, pBlock);
        modelFPType* sp = pBlock.getBlockPtr();
        if((last - first) == _nFeatures*nVectorsBeta)
        {
            setInterceptFlag(false);
            size_t i = 0;
            while(first != last)
            {
                if((i % (_nFeatures + 1)) == 0)
                {
                    sp[i] = 0;
                    ++i;
                }
                sp[i] = *first;
                ++first;
                ++i;
            }
        }
        else if((last - first) == (_nFeatures + 1)*nVectorsBeta)
        {
            setInterceptFlag(true);
            while(first != last)
            {
                *sp = *first;
                ++first;
                ++sp;
            }
        }
        else
        {
            _s = services::Status(services::ErrorIncorrectParameter);
            services::throwIfPossible(_s);
        }
        _modelPtr->getBeta()->releaseBlockOfRows(pBlock);
    }

    /**
     *  Get built model
     *  \return Model pointer
     */
    ModelPtr getModel()
    {
        return _modelPtr;
    }

    /**
     *  Get status of model building
     *  \return Status
     */
    services::Status getStatus()
    {
        return _s;
    }

private:
    ModelPtr _modelPtr;
    services::Status _s;
    size_t _nFeatures;
    size_t _nClasses;

    void setInterceptFlag(bool interceptFlag);
};

/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
