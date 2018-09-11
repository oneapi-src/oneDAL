/* file: bzip2compression.h */
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

/*
//++
//  Implementation of the bzip2 compression and decompression interface.
//--
*/

#ifndef __BZIP2COMPRESSION_H__
#define __BZIP2COMPRESSION_H__
#include "data_management/compression/compression.h"

namespace daal
{
/**
 * @defgroup data_management Data Management
 * \copydoc daal::data_management
 * @{
 */
namespace data_management
{

namespace interface1
{
/**
 * @defgroup data_compression Data Compression
 * \brief Contains classes for data compression and decompression
 * @ingroup data_management
 * @{
 */
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__BZIP2COMPRESSIONPARAMETER"></a>
 *
 * \brief Parameter for bzip2 compression and decompression
 *
 * \snippet compression/bzip2compression.h Bzip2CompressionParameter source code
 *
 * \par Enumerations
 *      - \ref CompressionLevel - %Compression level
 */
/* [Bzip2CompressionParameter source code] */
class DAAL_EXPORT Bzip2CompressionParameter : public data_management::CompressionParameter
{
public:
    /**
     *  Bzip2CompressionParameter Constructor
     *  \param[in] clevel   %Compression level, \ref CompressionLevel.
     *                      defaultLevel is equal to bzip2 compression level 9
     */
    Bzip2CompressionParameter( CompressionLevel clevel = defaultLevel ) :
        data_management::CompressionParameter(clevel)
    {}
    ~Bzip2CompressionParameter() {}
};
/* [Bzip2CompressionParameter source code] */

/**
 * <a name="DAAL-CLASS-COMPRESSOR_BZIP2"></a>
 *
 * \brief Implementation of the Compressor class for the bzip2 compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref Bzip2CompressionParameter class
 */
template<> class DAAL_EXPORT Compressor<bzip2> : public data_management::CompressorImpl
{
public:
    /**
     * \brief Compressor<bzip2> constructor
     */
    Compressor();
    ~Compressor();
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Pointer to the data block to compress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to compress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for compression in inBlock
     */
    void setInputDataBlock( byte *inBlock, size_t size, size_t offset );
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Reference to the data block to compress
     */
    void setInputDataBlock( DataBlock &inBlock )
    {
        return setInputDataBlock( inBlock.getPtr(), inBlock.getSize(), 0 );
    }
    /**
     * Performs bzip2 compression of a data block
     * \param[out] outBlock Pointer to the data block where compression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for compression in outBlock
     */
    void run( byte *outBlock, size_t size, size_t offset );
    /**
     * Performs bzip2 compression of a data block
     * \param[out] outBlock Reference to the data block where compression results are stored
     */
    void run( DataBlock &outBlock )
    {
        run( outBlock.getPtr(), outBlock.getSize(), 0 );
    }

    Bzip2CompressionParameter parameter; /*!< Bzip2 compression parameters structure */

protected:
    void initialize();

private:
    void *_strmp;
    int _flush;
    int _blockSize100k;
    size_t _comprLen;
    size_t _comprLenLeft;
    size_t _comprBlockThres;
    byte * _startAddr;

    void finalizeCompression();
    void resetCompression();
    void checkBZipError(int error);
};

/**
 * <a name="DAAL-CLASS-DECOMPRESSOR_BZI2"></a>
 *
 * \brief Specialization of Decompressor class for Bzip2 compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref Bzip2CompressionParameter class
 */
template<> class DAAL_EXPORT Decompressor<bzip2> : public data_management::DecompressorImpl
{
public:
    /**
     * \brief Decompressor<bzip2> constructor
     */
    Decompressor();
    ~Decompressor();
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Pointer to the data block to decompress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to decompress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for decompression in inBlock
     */
    void setInputDataBlock( byte *inBlock, size_t size, size_t offset );
    /**
     * Associates an input data block with a decompressor
     * \param[in] inBlock Reference to the data block to decompress
     */
    void setInputDataBlock( DataBlock &inBlock )
    {
        setInputDataBlock( inBlock.getPtr(), inBlock.getSize(), 0 );
    }
    /**
     * Performs bzip2 decompression of a data block
     * \param[out] outBlock Pointer to the data block where decompression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for decompression in outBlock
     */
    void run( byte *outBlock, size_t size, size_t offset );
    /**
     * Performs bzip2 decompression of a data block
     * \param[out] outBlock Reference to the data block where decompression results are stored
     */
    void run( DataBlock &outBlock )
    {
        run( outBlock.getPtr(), outBlock.getSize(), 0 );
    }

    Bzip2CompressionParameter parameter; /*!< Bzip2 compression parameters structure */

protected:
    void initialize();

private:
    void *_strmp;

    void finalizeCompression();
    void resetCompression();
    void checkBZipError(int error);

};
} // namespace interface1
using interface1::Bzip2CompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
/** @} */
} //namespace daal
#endif //__BZIP2COMPRESSION_H
