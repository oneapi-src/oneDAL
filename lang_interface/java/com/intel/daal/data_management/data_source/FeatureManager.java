/* file: FeatureManager.java */
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA_SOURCE__FEATUREMANAGER"></a>
 */
public class FeatureManager extends ContextClient {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    FeatureManager(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    public void setDelimiter(char delimiter) {
        cSetDelimiter(cObject, delimiter);
    }

    private native void cSetDelimiter(long cObject, char delimiter);

    public void addModifier(ModifierIface modifier) {
        cAddModifier(cObject, modifier.getCObject());
    }

    private native void cAddModifier(long cObject, long cModifier);

    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    private native void cDispose(long cObject);

    protected long cObject;
}
/** @} */
