/* file: ResultCollection.java */
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
 * @ingroup quality_metric
 * @{
 */
package com.intel.daal.algorithms.quality_metric_set;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Result;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 *  @brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 */
public class ResultCollection extends com.intel.daal.data_management.data.KeyValueDataCollection {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public ResultCollection(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public ResultCollection(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, (long)0);
        this.cObject = cInit(cAlgorithm);
    }

    /**
     * Adds an element with a key to the collection
     * @param key     Key value
     * @param value   %Result object
     */
    public void add(int key, Result value) {
        cAddResult(getCObject(), key, value.getCObject());
    }

    private native long cInit(long algaddr);

    private native void cAddResult(long collAddr, int key, long resAddr);

    protected native long cGetResult(long collAddr, int id);
}
/** @} */
