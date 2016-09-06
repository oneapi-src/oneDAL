/* file: AOSNumericTableImpl.java */
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

import java.lang.reflect.Field;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__AOSNUMERICTABLEIMPL"></a>
 * @brief Class that provides methods to access data that is stored as a contiguous array
 *         of heterogeneous feature vectors,  and each feature vector is represented
 *         with a data structure.
 *         Therefore, the data is represented as an Array Of Structures(AOS).
 */
public class AOSNumericTableImpl extends NumericTableImpl {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructor for empty Numeric Table with predefined class for the feature vectors and given number of feature vectors
     *
     * @param context   Context to manage created AOS numeric table
     * @param cls       Class containing expected array elements
     * @param nVectors  The number of rows in the table
     */
    public AOSNumericTableImpl(DaalContext context, Class<?> cls, long nVectors) {
        super(context);
        this.objClass = cls;
        this.fields = this.objClass.getFields();

        nJavaVectors = nVectors;
        nJavaFeatures = cls.getFields().length;
        dataAllocatedInJava = true;

        this.cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, NumericTable.StorageLayout.aos);

        dict = new DataDictionary(getContext(), nJavaFeatures, cGetCDataDictionary(cObject));
        initDataDictionary();
    }

    /**
     * Constructs Numeric Table from the array of objects representing feature vectors
     *
     * @param context Context to manage created AOS numeric table
     * @param ptr     Array of objects to associate with the Numeric Table
     */
    public AOSNumericTableImpl(DaalContext context, Object[] ptr) {
        super(context);
        this.objClass = ptr[0].getClass();
        this.fields = this.objClass.getFields();
        this.ptr = ptr;

        nJavaFeatures = this.objClass.getFields().length;
        nJavaVectors = ptr.length;
        dataAllocatedInJava = true;

        this.cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, NumericTable.StorageLayout.aos);

        dict = new DataDictionary(getContext(), nJavaFeatures, cGetCDataDictionary(cObject));
        initDataDictionary();
    }

    /**
     * Array of objects associated with the table
     */
    public void setArray(Object[] arr) {
        if (arr[0].getClass() != this.objClass) {
            throw new IllegalArgumentException("Incorrest class of Object");
        }
        this.ptr = arr;
        nJavaVectors = ptr.length;
        setNumberOfRows(nJavaVectors);
    }

    /**
     * Returns the array of objects associated with the table
     *
     * @return Array of objects
     */
    public Object[] getArray() {
        return this.ptr;
    }

    /**
     * Returns the data dictionary
     *
     * @return Data dictionary
     */
    public DataDictionary getDictionary() {
        return dict;
    }

    /**
     * Initializes Dictionary based on object class of known feature vectors
     */
    protected void initDataDictionary() {
        for (int i = 0; i < fields.length; i++) {

            Class<?> fCls = fields[i].getType();

            if (fCls == Long.TYPE) {
                dict.setFeature(Long.class, i);
            } else if (fCls == Integer.TYPE) {
                dict.setFeature(Integer.class, i);
            } else if (fCls == Float.TYPE) {
                dict.setFeature(Float.class, i);
            } else if (fCls == Double.TYPE) {
                dict.setFeature(Double.class, i);
            } else {
                throw new IllegalArgumentException("Incorrect feature class");
            }
        }
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    buf.put(fields[j].getDouble(this.ptr[(int) vectorIndex + i]));
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    double d = fields[j].getDouble(this.ptr[(int) vectorIndex + i]);
                    buf.put((float) d);
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfRows(long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    double d = fields[j].getDouble(this.ptr[(int) vectorIndex + i]);
                    buf.put((int) d);
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                buf.put(fields[(int) featureIndex].getDouble(this.ptr[(int) vectorIndex + i]));
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                double d = fields[(int) featureIndex].getFloat(this.ptr[(int) vectorIndex + i]);
                buf.put((float) d);
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                double d = fields[(int) featureIndex].getFloat(this.ptr[(int) vectorIndex + i]);
                buf.put((int) d);
            }
        } catch (java.lang.IllegalAccessException e) {
        }
        return buf;
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    fields[j].setDouble(this.ptr[(int) vectorIndex + i], buf.get());
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    fields[j].setFloat(this.ptr[(int) vectorIndex + i], buf.get());
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                for (int j = 0; j < fields.length; j++) {
                    fields[j].setInt(this.ptr[(int) vectorIndex + i], buf.get());
                }
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                fields[(int) featureIndex].setDouble(this.ptr[(int) vectorIndex + i], buf.get());
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                fields[(int) featureIndex].setFloat(this.ptr[(int) vectorIndex + i], buf.get());
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        try {
            for (int i = 0; i < vectorNum; i++) {
                fields[(int) featureIndex].setInt(this.ptr[(int) vectorIndex + i], buf.get());
            }
        } catch (java.lang.IllegalAccessException e) {
        }
    }

    protected transient Class<?> objClass;
    protected transient Field[]  fields;
    protected Object[]           ptr;

    @Override
    protected void onUnpack(DaalContext context) {
        if (dataAllocatedInJava) {
            this.objClass = this.ptr[0].getClass();
            this.fields = this.objClass.getFields();

            dataAllocatedInJava = true;

            this.cObject = newJavaNumericTable(nJavaFeatures, nJavaVectors, NumericTable.StorageLayout.aos);

            dict = new DataDictionary(getContext(), nJavaFeatures, cGetCDataDictionary(cObject));
            initDataDictionary();
        } else {
            super.onUnpack(context);
        }
    }
}
