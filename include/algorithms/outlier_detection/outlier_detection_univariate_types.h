/* file: outlier_detection_univariate_types.h */
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
//  univariate outlier detection algorithm types
//--
*/

#ifndef __OUTLIERDETECTION_UNIVARIATE_TYPES_H__
#define __OUTLIERDETECTION_UNIVARIATE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup univariate_outlier_detection Univariate Outlier Detection
 * \copydoc daal::algorithms::univariate_outlier_detection
 * @ingroup analysis
 * @{
 */
/**
* \brief Contains classes for computing results of the univariate outlier detection algorithm
*/
namespace univariate_outlier_detection
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__METHOD"></a>
 * Available methods for computing results of the univariate outlier detection algorithm
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__INPUTID"></a>
 * Available identifiers of input objects of the univariate outlier detection algorithm
 */
enum InputId
{
    data      , /*!< %Input data table */
    location  , /*!< Vector of mean estimates of size 1 x p */
    scatter   , /*!< Measure of spread, the array of standard deviations of size 1 x p */
    threshold , /*!< Limit that defines the outlier region, the array of non-negative numbers of size 1 x p */
    lastInputId = threshold
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__RESULTID"></a>
 * Available identifiers of results of the univariate outlier detection algorithm
 */
enum ResultId
{
    weights,           /*!< Table with results */
    lastResultId = weights
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__INPUT"></a>
 * \brief %Input objects for the univariate outlier detection algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns an input object for the univariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for the univariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks input objects for the univariate outlier detection algorithm
     * \param[in] par     Parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the univariate outlier detection algorithm in the %batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE();
    Result();

    virtual ~Result() {};

    /**
     * Registers user-allocated memory to store univariate outlier detection results
     * \param[in] input   Pointer to the %input objects for the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns a result of the univariate outlier detection algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets a result of the univariate outlier detection algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result object of the univariate outlier detection algorithm
     * \param[in] input   Pointer to the  %input objects for the algorithm
     * \param[in] par     Pointer to the parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
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

} // namespace univariate_outlier_detection
} // namespace algorithm
} // namespace daal
#endif
