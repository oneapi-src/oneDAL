/* file: Pooling1dParameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DPARAMETER"></a>
 * \brief Class that specifies parameters of the one-dimensional pooling layer
 */
public class Pooling1dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
