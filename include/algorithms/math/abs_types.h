/* file: abs_types.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of the absolute value function interface.
//--
*/

#ifndef __ABS_TYPES_H__
#define __ABS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup math Math Functions
 * \copydoc daal::algorithms::math
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for computing math functions
 */
namespace math
{
/**
 * @defgroup abs Absolute Value(abs)
 * \copydoc daal::algorithms::math::abs
 * @ingroup math
 * @{
 */
/**
 * \brief Contains classes for computing the absolute value function
 */
namespace abs
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ABS__METHOD"></a>
 * Computation methods for computing the absolute value function
 * @ingroup abs
 */
enum Method
{
    defaultDense = 0,       /*!< Default: performance-oriented method. */
    fastCSR = 1             /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows(CSR) numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ABS__INPUTID"></a>
 * Available identifiers of input objects for the absolute value function
 * @ingroup abs
 */
enum InputId
{
    data = 0               /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ABS__RESULTID"></a>
 * Available identifiers of the result of the absolute value function
 * @ingroup abs
 */
enum ResultId
{
    value = 0     /*!< Table to store the result */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__ABS__INPUT"></a>
 * \brief %Input objects for the absolute value function
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    virtual ~Input() {}

    /**
     * Returns an input object for the absolute value function
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for the absolute value function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks an input object for the absolute value function
     * \param[in] par     Function parameter
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__ABS__RESULT"></a>
 * \brief Result obtained with the compute() method of the absolute value function in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the result of the absolute value function
     * \param[in] input  %Input object for the absolute value function
     * \param[in] par    %Parameter of the absolute value function
     * \param[in] method Computation method of the absolute value function
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Returns the result of the absolute value function
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the absolute value function
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result of the absolute value function
     * \param[in] in   %Input of the absolute value function
     * \param[in] par     %Parameter of the absolute value function
     * \param[in] method  Computation method of the absolute value function
     */
    void check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns the serialization tag of the result
    * \return         Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_ABS_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
/** @} */
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;

} // namespace abs
} // namespace math
} // namespace algorithm
} // namespace daal
#endif
