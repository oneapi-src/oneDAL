/* file: outlier_detection_multivariate_types.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Outlier Detection algorithm parameter structure
//--
*/

#ifndef __OUTLIERDETECTION_MULTIVARIATE_TYPES_H__
#define __OUTLIERDETECTION_MULTIVARIATE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup multivariate_outlier_detection Multivariate Outlier Detection
 * \copydoc daal::algorithms::multivariate_outlier_detection
 * @ingroup analysis
 * @{
 */
/**
* \brief Contains classes for computing the multivariate outlier detection
*/
namespace multivariate_outlier_detection
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__METHOD"></a>
 * Available computation methods for the multivariate outlier detection algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INPUTID"></a>
 * Available identifiers of input objects for the multivariate outlier detection algorithm
 */
enum InputId
{
    data      , /*!< %Input data table */
    location  , /*!< Measure of spread, the variance-covariance matrix of size p x p */
    scatter   , /*!< Vector of mean estimates of size 1 x p */
    threshold , /*!< Limit that defines the outlier region, the array of size 1 x 1 containing a non-negative number */
    lastInputId = threshold
};


/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__RESULTID"></a>
 * Available identifiers of the results of the multivariate outlier detection algorithm
 */
enum ResultId
{
    weights, /*!< Outlier detection results */
    lastResultId = weights
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INPUT"></a>
 * \brief %Input objects for the multivariate outlier detection algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns input object for the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks input object for the multivariate outlier detection algorithm
     * \param[in] par     Algorithm parameters
     * \param[in] method  Computation method for the algorithm
     *
     * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the multivariate outlier detection algorithm in the %batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE();
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the multivariate outlier detection algorithm
     * \tparam algorithmFPType  Data type to use for storing results, double or float
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns result of the multivariate outlier detection algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result object of the multivariate outlier detection algorithm
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] par     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}
};
typedef services::SharedPtr<Result> ResultPtr;

/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace multivariate_outlier_detection
} // namespace algorithm
} // namespace daal
#endif
