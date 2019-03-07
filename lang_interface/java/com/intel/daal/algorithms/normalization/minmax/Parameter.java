/* file: Parameter.java */
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
 * @ingroup minmax
 * @{
 */
package com.intel.daal.algorithms.normalization.minmax;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.low_order_moments.BatchImpl;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__MINMAX__PARAMETER"></a>
 * @brief Parameters of the Min-max normalization algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of Min-max normalization algorithm
     *
     * @param context                  Context to manage the Min-max normalization algorithm
     * @param cObject                  Address of C++ parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets the moments algorithm used by Min-max normalization algorithm
     * @param moments Low order moments algorithm
     */
    public void setMoments(BatchImpl moments) {
        cSetMoments(this.cObject, moments.cBatchImpl);
    }

    /**
     * Gets the lower bound of the features value will be obtained during normalization.
     * @return  The lower bound of the features value will be obtained during normalization
     */
    public double getLowerBound() {
        return cGetLowerBound(this.cObject);
    }

    /**
     * Sets the lower bound of the features value will be obtained during normalization.
     * @param lowerBound The lower bound of the features value will be obtained during normalization
     */
    public void setLowerBound(double lowerBound) {
        cSetLowerBound(this.cObject, lowerBound);
    }

    /**
     * Gets the upper bound of the features value will be obtained during normalization.
     * @return The upper bound of the features value will be obtained during normalization
     */
    public double getUpperBound() {
        return cGetUpperBound(this.cObject);
    }

    /**
     * Sets the upper bound of the features value will be obtained during normalization.
     * @param upperBound The upper bound of the features value will be obtained during normalization
     */
    public void setUpperBound(double upperBound) {
        cSetUpperBound(this.cObject, upperBound);
    }


    private native void cSetMoments(long cObject, long moments);

    private native void cSetLowerBound(long cObject, double lowerBound);
    private native void cSetUpperBound(long cObject, double upperBound);

    private native double cGetLowerBound(long cObject);
    private native double cGetUpperBound(long cObject);
}
/** @} */
