/* file: QualityMetricInput.java */
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
package com.intel.daal.algorithms.quality_metric;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC__QUALITYMETRICINPUT"></a>
 * @brief  Base class for input objects of quality metrics
 */
public class QualityMetricInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the input of the quality metric algorithm
     * @param context   Context to manage the input of the quality metric algorithm
     */
    public QualityMetricInput(DaalContext context) {
        super(context);
    }

    public QualityMetricInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
