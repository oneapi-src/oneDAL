/* file: DataDictionary.java */
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

package com.intel.daal.data_management.data;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__DATADICTIONARY"></a>
 * @brief Class that represents the data set dictionary and provides
 *        methods to work with the data dictionary. Methods of the class use the
 *        com.intel.daal.data.DataFeature structure.
 */
public class DataDictionary extends SerializableBase {

    /**
     * Constructs the dictionary
     *
     * @param context        Context to manage created data set dictionary
     * @param nFeatures      The number of features in the table
     * @param featuresEqual  Flag specifying that all features have equal types and properties
     */
    public DataDictionary(DaalContext context, long nFeatures, boolean featuresEqual) {
        super(context);
        _dictionaryAllocatedInJava = true;
        _featuresEqual = featuresEqual;
        this.cObject = init((int) nFeatures, _featuresEqual);
        dict = new Vector<DataFeature>();
        setNumberOfFeatures(nFeatures);
    }

    /**
     * Constructs the dictionary
     *
     * @param context        Context to manage created data set dictionary
     * @param nFeatures      The number of features in the table
     */
    public DataDictionary(DaalContext context, long nFeatures) {
        super(context);
        _dictionaryAllocatedInJava = true;
        _featuresEqual = false;
        this.cObject = init((int) nFeatures, _featuresEqual);
        dict = new Vector<DataFeature>();
        setNumberOfFeatures(nFeatures);
    }

    /**
     * Constructs the dictionary from the defined C dictionary
     *
     * @param context        Context to manage created data set dictionary
     * @param nFeatures      The number of features in the table
     * @param cDictionary    Pointer to the C dictionary
     */
    public DataDictionary(DaalContext context, long nFeatures, long cDictionary) {
        super(context);
        cObject = cDictionary;
        _dictionaryAllocatedInJava = false;
        nFeatures = cGetNumberOfFeatures(cObject);
        _featuresEqual = cGetCDataDictionaryFeaturesEqual(cObject);
        dict = new Vector<DataFeature>();
        if (_featuresEqual) {
            dict.setSize(1);
        } else {
            dict.setSize((int) nFeatures);
        }
        for (int i = 0; i < dict.size(); i++) {
            DataFeature feature = new DataFeature(getContext());
            int indexType = cGetIndexType(cObject, i);
            if(indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT32.getType()) {
                feature.setType(Float.class);
            }
            else if(indexType == DataFeatureUtils.IndexNumType.DAAL_FLOAT64.getType()) {
                feature.setType(Double.class);
            }
            else if(indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_S.getType() ||
                    indexType == DataFeatureUtils.IndexNumType.DAAL_INT64_U.getType()) {
                feature.setType(Long.class);
            }
            else if(indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_S.getType() ||
                    indexType == DataFeatureUtils.IndexNumType.DAAL_INT32_U.getType()) {
                feature.setType(Integer.class);
            }
            dict.set(i, feature);
        }
    }

    /**
     * Sets number of features
     *
     * @param nFeatures      Number of features in the table
     */
    public void setNumberOfFeatures(long nFeatures) {
        if (_featuresEqual) {
            dict.setSize(1);
        }
        else {
            dict.setSize((int) nFeatures);
        }
        cSetNumberOfFeatures(getCObject(), nFeatures);
        serializedNFeatures = nFeatures;
    }

    /**
     * Returns the number of features
     *
     * @return Number of features
     */
    public int getNumberOfFeatures() {
        return cGetNumberOfFeatures(getCObject());
    }

    /**
     *  Resets dictionary and sets number of features to 0
     */
    public void resetDictionary() {
        dict.clear();
        cResetDictionary(getCObject());
    }

    /**
     *  Sets all features of the dictionary to the same type
     *  @param defaultFeature  Default feature class to set all features to
     */
    public void setAllFeatures(DataFeature defaultFeature) {
        for (int i = 0; i < dict.size(); i++) {
            dict.set(i, defaultFeature);
        }
        cSetAllFeatures(getCObject(), defaultFeature.getCObject());
    }

    /**
     * Sets a feature in the data dictionary
     *
     * @param feature        Data feature
     * @param idx            Index of the data feature
     */
    public void setFeature(DataFeature feature, int idx) {
        if (_featuresEqual) {
            dict.set(0, feature);
        } else {
            dict.set(idx, feature);
        }
        cSetFeature(getCObject(), feature.getCObject(), idx);
    }

    /**
     * Sets a feature in the data dictionary
     *
     * @param cls            Class containing values of the data feature
     * @param idx            Index of the data feature
     * @param featureType    Feature type
     * @param categoryNumber Number of categories
     */
    public void setFeature(Class<?> cls, int idx, DataFeatureUtils.FeatureType featureType, int categoryNumber) {
        DataFeature df = new DataFeature(getContext());
        df.setType(cls);
        df.setFeatureType(featureType);
        df.setCategoryNumber(categoryNumber);
        setFeature(df, idx);
    }

    /**
     * Sets a feature in the data dictionary
     *
     * @param cls           Class containing values of the data feature
     * @param idx           Index of the data feature
     * @param featureType   Feature type
     */
    public void setFeature(Class<?> cls, int idx, DataFeatureUtils.FeatureType featureType) {
        setFeature(cls, idx, featureType, 0);
    }

    /**
     * Sets a feature in the data dictionary
     *
     * @param cls Class containing values of the data feature
     * @param idx Index of the data feature
     */
    public void setFeature(Class<?> cls, int idx) {
        setFeature(cls, idx, DataFeatureUtils.FeatureType.DAAL_CONTINUOUS, 0);
    }

    /**
     * Returns the feature with a given index
     *
     * @param idx Index of the data feature
     *
     * @return Feature with the given index.
     */
    public DataFeature getFeature(int idx) {
        if (_featuresEqual) {
            return dict.get(0);
        } else {
            return dict.get(idx);
        }
    }

    /* Constructs C++ data dictionary object */
    private native long init(int nFeatures, boolean featuresEqual);

    /* Sets number of features in the dictionary */
    private native void cSetNumberOfFeatures(long cObject, long nFeatures);

    /* Gets number of features in the dictionary */
    private native int cGetNumberOfFeatures(long cObject);

    /* Sets a feature in the data dictionary */
    private native void cSetFeature(long cObject, long featureAddr, int idx);

    /* Sets all features to the same type */
    private native void cSetAllFeatures(long cObject, long featureAddr);

    /* Reset dictionary */
    private native void cResetDictionary(long cObject);

    /* Gets the value of the dictionary featuresEqual flag */
    private native boolean cGetCDataDictionaryFeaturesEqual(long cObject);

    /* Gets type of the feature from C dictionary  */
    private native int cGetIndexType(long cObject, int idx);

    @Override
    protected void onUnpack(DaalContext context) {
        deserializeCObject();

        if (_dictionaryAllocatedInJava) {
            cObject = init((int) serializedNFeatures, _featuresEqual);
        }
    }

    /* Implementation of the deserialization function */
    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
        serializedNFeatures = aInputStream.readLong();
    }

    /* Implementation of the serialization function */
    private void writeObject(ObjectOutputStream aOutputStream) throws IOException {
        aOutputStream.defaultWriteObject();
        aOutputStream.writeLong(serializedNFeatures);
    }

    /** @private */
    Vector<DataFeature> dict;

    transient byte[]       serializedCObject;
    private transient long serializedNFeatures;
    private boolean        _featuresEqual;
    private boolean        _dictionaryAllocatedInJava;
}
