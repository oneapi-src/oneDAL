/* file: ColumnFilter.java */
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__COLUMNFILTER"></a>
 */
public class ColumnFilter extends ModifierIface {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ColumnFilter(DaalContext context) {
        super(context);
        this.cObject = cInit();
    }

    public ColumnFilter odd() {
        cOdd(cObject);
        return this;
    }

    public ColumnFilter even() {
        cEven(cObject);
        return this;
    }

    public ColumnFilter none() {
        cNone(cObject);
        return this;
    }

    public ColumnFilter list(long[] valid) {
        cList(cObject, valid);
        return this;
    }

    private native long cInit();
    private native void cOdd(long cObject);
    private native void cEven(long cObject);
    private native void cNone(long cObject);
    private native void cList(long cObject, long[] valid);
}
/** @} */
