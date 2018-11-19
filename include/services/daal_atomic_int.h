/* file: daal_atomic_int.h */
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
//  Declaration of class for atomic operations with int
//--
*/

#ifndef __DAAL_ATOMIC_INT_H__
#define __DAAL_ATOMIC_INT_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"

namespace daal
{
namespace services
{

namespace interface1
{
/**
 * @ingroup memory
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__ATOMIC"></a>
 * \brief Class that represents an atomic object
 *
 * \tparam dataType Data type of the atomic object
 */
template<typename dataType = int>
class DAAL_EXPORT Atomic
{
public:
    /**
     * Returns an increment of atomic object
     * \return An increment of atomic object
     */
    dataType inc();

    /**
     * Returns a decrement of atomic object
     * \return An decrement of atomic object
     */
    dataType dec();

    /**
     * Assigns the value to atomic object
     * \param[in] value    The value to be assigned
     */
    void set(dataType value);

    /**
     * Returns the value of the atomic object
     * \return The value of the atomic object
     */
    dataType get() const;

    /**
     * Constructs an atomic object
     */
    Atomic();

    /**
     * Constructs an atomic object from a value
     * \param[in] value The value to be assigned to the atomic object
     */
    Atomic(dataType value);

    /** Destructor */
    ~Atomic();

protected:
    void *_ptr;

private:
    Atomic(const Atomic &);
};


/** @} */

} // namespace interface1
using interface1::Atomic;

typedef Atomic<int> AtomicInt;

}
}

#endif
