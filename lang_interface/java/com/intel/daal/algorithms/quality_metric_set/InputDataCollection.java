/* file: InputDataCollection.java */
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
import com.intel.daal.algorithms.Input;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 *  @brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 */
public class InputDataCollection extends com.intel.daal.data_management.data.KeyValueDataCollection {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InputDataCollection(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public InputDataCollection(DaalContext context, long cAlgorithm, ComputeMode cmode) {
        super(context, (long)0);
        this.cObject = cInit(cAlgorithm);
    }

    /**
     * Adds an element with a key to the collection
     * @param key     Key value
     * @param value   Input object
     */
    public void add(int key, Input value) {
        cAddInput(getCObject(), key, value.getCObject());
    }

    private native long cInit(long algaddr);

    private native void cAddInput(long collAddr, int key, long inputAddr);

    protected native long cGetInput(long collAddr, int id);

    /**
     * Releases memory allocated for the native parameter object
     */
    @Override
    public void dispose() {
        /* Will be disposed with owning QualityMetricSetBatch */ }
}
/** @} */
