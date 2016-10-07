/* file: LocallyConnected2dParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DPARAMETER"></a>
 * \brief Class that specifies parameters of the 2D locally connected layer
 */
public class LocallyConnected2dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    public LocallyConnected2dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     *  Gets the index of the dimension for which the grouping is applied
     * @return LocallyConnected2dIndex of the dimension for which the grouping is applied
     */
    public long getGroupDimension() {
        return cGetGroupDimension(cObject);
    }

    /**
     *  Sets the index of the dimension for which the grouping is applied
     *  @param groupDimension   Dimension for which the grouping is applied
     */
    public void setGroupDimension(long groupDimension) {
        cSetGroupDimension(cObject, groupDimension);
    }

    /**
    *  Gets the number of kernels applied to the input layer data
    * @return Number of kernels applied to the input layer data
    */
    public long getNKernels() {
        return cGetNKernels(cObject);
    }

    /**
     *  Sets the number of kernels applied to the input layer data
     *  @param nKernels   The number of kernels applied to the input layer data
     */
    public void setNKernels(long nKernels) {
        cSetNKernels(cObject, nKernels);
    }

    /**
     *  Gets the number of groups which the input data is split in groupDimension dimension
     * @return Number of groups which the input data is split in groupDimension dimension
     */
    public long getNGroups() {
        return cGetNGroups(cObject);
    }

    /**
     *  Sets the number of groups which the input data is split in groupDimension dimension
     *  @param nGroups   The number of groups which the input data is split in groupDimension dimension
     */
    public void setNGroups(long nGroups) {
        cSetNGroups(cObject, nGroups);
    }

    /**
     * Gets the data structure representing the sizes of the two-dimensional kernel subtensor
     * @return Data structure representing the sizes of the two-dimensional kernel subtensor
     */
    public LocallyConnected2dKernelSizes getKernelSizes() {
        long[] size = cGetKernelSizes(cObject);
        return new LocallyConnected2dKernelSizes(size[0], size[1]);
    }

    /**
     * Sets the data structure representing the sizes of the two-dimensional kernel subtensor
     * @param kernelSize   The data structure representing the sizes of the two-dimensional kernel subtensor
     */
    public void setKernelSizes(LocallyConnected2dKernelSizes kernelSize) {
        long[] size = kernelSize.getSize();
        cSetKernelSizes(cObject, size[0], size[1]);
    }

    /**
     * Gets the data structure representing the intervals on which the kernel should be applied to the input
     * @return Data structure representing the intervals on which the kernel should be applied to the input
     */
    public LocallyConnected2dStrides getStrides() {
        long[] size = cGetStrides(cObject);
        return new LocallyConnected2dStrides(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the intervals on which the kernel should be applied to the input
     *  @param str   The data structure representing the intervals on which the kernel should be applied to the input
     */
    public void setStrides(LocallyConnected2dStrides str) {
        long[] size = str.getSize();
        cSetStrides(cObject, size[0], size[1]);
    }

    /**
     * Gets the data structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which locally connected is performed
     * @return Data structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which locally connected is performed
     */
    public LocallyConnected2dPaddings getPaddings() {
        long[] size = cGetPaddings(cObject);
        return new LocallyConnected2dPaddings(size[0], size[1]);
    }

    /**
     * Sets the data structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which locally connected is performed
     * @param padding   The data structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which locally connected is performed
     */
    public void setPaddings(LocallyConnected2dPaddings padding) {
        long[] size = padding.getSize();
        cSetPaddings(cObject, size[0], size[1]);
    }

    /**
     * Gets the data structure representing the dimension for locally connected kernels
     * @return Data structure representing the dimension for locally connected kernels
     */
    public LocallyConnected2dIndices getIndices() {
        long[] size = cGetIndices(cObject);
        return new LocallyConnected2dIndices(size[0], size[1]);
    }

    /**
     * Sets the data structure representing the dimension for locally connected kernels
     * @param LocallyConnected2dIndices   The data structure representing the dimension for locally connected kernels
     */
    public void setIndices(LocallyConnected2dIndices LocallyConnected2dIndices) {
        long[] size = LocallyConnected2dIndices.getSize();
        cSetIndices(cObject, size[0], size[1]);
    }

    private native long cInit();
    private native long cGetGroupDimension(long cObject);
    private native void cSetGroupDimension(long cObject, long groupDimension);
    private native long cGetNKernels(long cObject);
    private native void cSetNKernels(long cObject, long nKernels);
    private native long cGetNGroups(long cObject);
    private native void cSetNGroups(long cObject, long nGroups);
    private native void cSetKernelSizes(long cObject, long first, long second);
    private native void cSetStrides(long cObject, long first, long second);
    private native void cSetPaddings(long cObject, long first, long second);
    private native void cSetIndices(long cObject, long first, long second);
    private native long[] cGetKernelSizes(long cObject);
    private native long[] cGetStrides(long cObject);
    private native long[] cGetPaddings(long cObject);
    private native long[] cGetIndices(long cObject);
}
