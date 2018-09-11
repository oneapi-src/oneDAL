/** file error_id.h */
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
//  Data types for error handling in Intel(R) DAAL.
//--
*/

#ifndef __ERROR_ID__
#define __ERROR_ID__

#include "error_indexes.h"
#include "services/daal_defines.h"
#include "daal_memory.h"

namespace daal
{
namespace services
{

namespace interface1
{
/**
 * @ingroup error_handling
 * @{
 */
/**
* <a name="DAAL-CLASS-SERVICES__ERRORDETAIL"></a>
* \brief Base for error detail classes
*/
class DAAL_EXPORT ErrorDetail
{
public:
    DAAL_NEW_DELETE();

    /**
    * Constructs error detail from error identifier
    * \param[in] id    Error identifier
    */
    ErrorDetail(ErrorDetailID id) : _id(id), _next(NULL){}

    /**
    * Destructor
    */
    virtual ~ErrorDetail(){}

    /**
    * Returns identifier of an error detail
    * \return identifier of an error detail
    */
    ErrorDetailID id() const { return _id; }

    /**
    * Returns copy of this object
    * \return copy of this object
    */
    virtual ErrorDetail* clone() const = 0;

    /**
    * Adds description of the error detail to the given string
    * \param[in] str String to add descrition to
    */
    virtual void describe(char* str) const = 0;

    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    const ErrorDetail* next() const { return _next; }

protected:
    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    ErrorDetail* next() { return _next; }

    /**
    * Set pointer to the next detail
    * \param[in] ptr Pointer to the next detail
    */
    void addNext(ErrorDetail* ptr) { _next = ptr; }

private:
    const ErrorDetailID _id;
    ErrorDetail* _next;
    friend class Error;
};
/** @} */

} // namespace interface1
using interface1::ErrorDetail;

}
}
#endif
