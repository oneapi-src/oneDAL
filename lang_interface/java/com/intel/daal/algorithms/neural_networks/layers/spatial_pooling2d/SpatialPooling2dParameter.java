/* file: SpatialPooling2dParameter.java */
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
