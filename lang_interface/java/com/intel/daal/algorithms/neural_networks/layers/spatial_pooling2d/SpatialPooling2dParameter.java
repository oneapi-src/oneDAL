/* file: SpatialPooling2dParameter.java */
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
 * @ingroup spatial_pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_POOLING2D__SPATIALPOOLING2DPARAMETER"></a>
 * \brief Class that specifies parameters of the two-dimensional spatial pooling layer
 */
public class SpatialPooling2dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    public SpatialPooling2dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    *  Gets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
    * @return Data structure representing the indices of the dimension on which two-dimensional pooling is performed
    */
    public SpatialPooling2dIndices getIndices() {
        long[] size = cGetIndices(cObject);
        return new SpatialPooling2dIndices(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
     *  @param sd   The data structure representing the indices of the dimension on which two-dimensional pooling is performed
     */
    public void setIndices(SpatialPooling2dIndices sd) {
        long[] size = sd.getSize();
        cSetIndices(cObject, size[0], size[1]);
    }

    /**
     *  Returns pyramidHeight for multinomial distribution random number generator
     */
    public long getPyramidHeight() {
        return cGetPyramidHeight(cObject);
    }

    /**
     *  Sets the pyramidHeight for multinomial distribution random number generator
     *  @param pyramidHeight PyramidHeight for multinomial distribution random number generator
     */
    public void setPyramidHeight(long pyramidHeight) {
        cSetPyramidHeight(cObject);
    }

    private native void cSetIndices(long cObject, long first, long second);
    private native long[] cGetIndices(long cObject);
    private native long   cGetPyramidHeight(long cParameter);
    private native void   cSetPyramidHeight(long cParametert);

}
/** @} */
