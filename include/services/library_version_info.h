/* file: library_version_info.h */
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
//  Intel(R) DAAL version information.
//--
*/

#ifndef __LIBRARY_VERSION_INFO_H__
#define __LIBRARY_VERSION_INFO_H__


#define __INTEL_DAAL_BUILD_DATE 21990101

#define __INTEL_DAAL__          2199
#define __INTEL_DAAL_MINOR__    9
#define __INTEL_DAAL_UPDATE__   9

#define INTEL_DAAL_VERSION (__INTEL_DAAL__ * 10000 + __INTEL_DAAL_MINOR__ * 100 + __INTEL_DAAL_UPDATE__)


#include "services/base.h"

namespace daal
{
/**
 * @defgroup services Services
 * \copydoc daal::services
 * @{
 */
namespace services
{

namespace interface1
{
/**
 * @defgroup library_version_info Extracting Version Information
 * \brief Provides information about the version of Intel(R) Data Analytics Acceleration Library.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__LIBRARYVERSIONINFO"></a>
 * \brief Provides information about the version of Intel(R) Data Analytics Acceleration Library.
 * <!-- \n<a href="DAAL-REF-LIBRARYVERSIONINFO-STRUCTURE">LibraryVersionInfo structure details and Optimization Notice</a> -->
 */
class DAAL_EXPORT LibraryVersionInfo: public Base
{
public:
    const int    majorVersion;   /*!< Major library version */
    const int    minorVersion;   /*!< Minor library version */
    const int    updateVersion;  /*!< Update library version */
    const char *productStatus;   /*!< Product library status */
    const char *build;           /*!< Library build */
    const char *build_rev;       /*!< Library build revision */
    const char *name;            /*!< Library name */
    const char *processor;       /*!< Instruction set supported by the processor */

    LibraryVersionInfo();
    ~LibraryVersionInfo();
};
/** @} */
} // namespace interface1
using interface1::LibraryVersionInfo;

}
/** @} */
}
#endif
