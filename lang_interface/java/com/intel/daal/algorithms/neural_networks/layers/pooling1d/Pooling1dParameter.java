/* file: Pooling1dParameter.java */
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

/**
 * @ingroup pooling1d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DPARAMETER"></a>
 * \brief Class that specifies parameters of the one-dimensional pooling layer
 */
public class Pooling1dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public Pooling1dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets the data structure representing the size of the 1D subtensor from which the element is computed
     * @return Data structure representing the size of the 1D subtensor from which the element is computed
     */
    public Pooling1dKernelSize getKernelSize() {
        long[] size = cGetKernelSize(cObject);
        return new Pooling1dKernelSize(size[0]);
    }

    /**
     *  Sets the data structure representing the size of the 1D subtensor from which the element is computed
     *  @param ks   The data structure representing the size of the 1D subtensor from which the element is computed
     */
    public void setKernelSize(Pooling1dKernelSize ks) {
        long[] size = ks.getSize();
        cSetKernelSize(cObject, size[0]);
    }

    /**
    *  Gets the data structure representing the intervals on which the subtensors for one-dimensional pooling are computed
    * @return Data structure representing the intervals on which the subtensors for one-dimensional pooling are selected
    */
    public Pooling1dStride getStride() {
        long[] size = cGetStride(cObject);
        return new Pooling1dStride(size[0]);
    }

    /**
     *  Sets the data structure representing the intervals on which the subtensors for one-dimensional pooling are selected
     *  @param str   The data structure representing the intervals on which the subtensors for one-dimensional pooling are selected
     */
    public void setStride(Pooling1dStride str) {
        long[] size = str.getSize();
        cSetStride(cObject, size[0]);
    }

    /**
    *  Gets the structure representing the number of data elements to implicitly add
    *        to each side of the 1D subtensor on which one-dimensional pooling is performed
    * @return Data structure representing the number of data elements to implicitly add to each size
    *         of the one-dimensional subtensor on which one-dimensional pooling is performed
    */
    public Pooling1dPadding getPadding() {
        long[] size = cGetPadding(cObject);
        return new Pooling1dPadding(size[0]);
    }

    /**
    *  Sets the data structure representing the number of data elements to implicitly add to each size
    *  of the one-dimensional subtensor on which one-dimensional pooling is performed
    *  @param pad   The data structure representing the number of data elements to implicitly add to each size
    *                      of the one-dimensional subtensor on which one-dimensional pooling is performed
    */
    public void setPadding(Pooling1dPadding pad) {
        long[] size = pad.getSize();
        cSetPadding(cObject, size[0]);
    }

    /**
    *  Gets the data structure representing the indices of the dimension on which one-dimensional pooling is performed
    * @return Data structure representing the indices of the dimension on which one-dimensional pooling is performed
    */
    public Pooling1dIndex getIndex() {
        long[] size = cGetSD(cObject);
        return new Pooling1dIndex(size[0]);
    }

    /**
     *  Sets the data structure representing the indices of the dimension on which one-dimensional pooling is performed
     *  @param sd   The data structure representing the indices of the dimension on which one-dimensional pooling is performed
     */
    public void setIndex(Pooling1dIndex sd) {
        long[] size = sd.getSize();
        cSetSD(cObject, size[0]);
    }

    private native void cSetKernelSize(long cObject, long first);
    private native void cSetStride(long cObject, long first);
    private native void cSetPadding(long cObject, long first);
    private native void cSetSD(long cObject, long first);
    private native long[] cGetKernelSize(long cObject);
    private native long[] cGetStride(long cObject);
    private native long[] cGetPadding(long cObject);
    private native long[] cGetSD(long cObject);
}
/** @} */
