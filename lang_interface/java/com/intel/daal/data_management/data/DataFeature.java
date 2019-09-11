/* file: DataFeature.java */
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
 * @ingroup data_dictionary
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import java.nio.ByteBuffer;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__DATAFEATURE"></a>
 * @brief Class used to describe a feature. The structure is used in the
 *        com.intel.daal.data.DataDictionary class.
 */
public class DataFeature extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the data feature
     * @param context   Context to manage the data feature
     */
    public DataFeature(DaalContext context) {
        super(context);
        this.cObject = init();
    }

    /**
     * Sets PMML data type of the feature
     * @param pmmlType      PMML data type of the feature
     */
    public void setPMMLNumType(DataFeatureUtils.PMMLNumType pmmlType) {
        cSetPMMLNumType(this.cObject, pmmlType.getType());
    }

    /**
     * Sets  type of the feature(continuous, ordinal or categorical)
     * @param featureType   Type of the feature
     */
    public void setFeatureType(DataFeatureUtils.FeatureType featureType) {
        cSetFeatureType(this.cObject, featureType.getType());
    }

    /**
     * Sets number of category levels
     * @param categoryNumber    Number of category levels
     */
    public void setCategoryNumber(int categoryNumber) {
        cSetCategoryNumber(this.cObject, categoryNumber);
    }

    /**
     * @private
     */
    void setType(Class<?> cls) {
        if (Double.class == cls || double.class == cls) {
            type = Double.class;
            cSetDoubleType(this.cObject);
        } else if (Float.class == cls || float.class == cls) {
            type = Float.class;
            cSetFloatType(this.cObject);
        } else if (Long.class == cls || long.class == cls) {
            type = Long.class;
            cSetLongType(this.cObject);
        } else if (Integer.class == cls || int.class == cls) {
            type = Integer.class;
            cSetIntType(this.cObject);
        } else {
            throw new IllegalArgumentException("type unsupported");
        }
    }

    /**
     * Gets PMML data type of the feature
     *
     * @return PMML data type of the feature
     */
    public native DataFeatureUtils.PMMLNumType getPMMLNumType();

    /**
     * Gets  type of the feature(continuous, ordinal or categorical)
     *
     * @return Type of the feature
     */
    public native DataFeatureUtils.FeatureType getFeatureType();

    /**
     * Gets number of category levels
     *
     * @return Number of category levels
     */
    public native int getCategoryNumber();

    /* Constructs C++ data feature object */
    private native long init();

    private native void cSetInternalNumType(long cObject, int intType);

    private native void cSetPMMLNumType(long cObject, int pmmlType);

    private native void cSetFeatureType(long cObject, int featureType);

    private native void cSetCategoryNumber(long cObject, int categoryNumber);

    private native void cSetName(long cObject, String name);

    private native void cSetDoubleType(long cObject);

    private native void cSetFloatType(long cObject);

    private native void cSetLongType(long cObject);

    private native void cSetIntType(long cObject);

    private native ByteBuffer cSerializeCObject(long cObject);
    private native long cDeserializeCObject(ByteBuffer buffer, long size);

    /** @private */
    Class<?> type;
}
/** @} */
