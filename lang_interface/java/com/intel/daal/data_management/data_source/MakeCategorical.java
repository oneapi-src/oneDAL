/* file: MakeCategorical.java */
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

/**
 * @defgroup data_sources Data Sources
 * @brief Specifies methods to access data
 * @ingroup data_management
 * @{
 */
/**
 */
package com.intel.daal.data_management.data_source;

import com.intel.daal.utils.*;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__MAKECATEGORICAL"></a>
 */
public class MakeCategorical extends ModifierIface {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MakeCategorical(DaalContext context, long idx) {
        super(context);
        this.cObject = cInit(idx);
    }

    private native long cInit(long idx);
}
/** @} */
