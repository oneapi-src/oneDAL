.. ******************************************************************************
.. * Copyright 2023 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/


Deprecation Notice
==================

This page provides information about the deprecations of a specific oneDAL functionality. 

Java* Interfaces
****************

**Deprecation:** The Java* interfaces in the oneDAL library are marked as deprecated. The future releases of the oneDAL library may no longer include support for these Java* interfaces.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features. 

**Alternatives:** Intel(R) Optimized Analytics Package* (OAP) project for the Spark* users. 
The project offers a comprehensive set of optimized libraries, including the OAP* MLlib* component. For more information, visit https://github.com/oap-project/oap-mllib. 


DAAL CPP SYCL Interfaces
****************

**Deprecation:** The DAAL CPP SYCL Interfaces(examples/daal/cpp_sycl) in the oneDAL library are marked as deprecated. From 2024.0 release of the oneDAL library will no longer include support for these DAAL CPP SYCL Interfaces.

**Reasons for deprecation:** Deprecating initial version of SYCL interfaces in favour of oneDAL SYCL interfaces. The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features. 

**Alternatives:** Use oneDAL SYCL interfaces(examples/oneapi/dpc) instead.


Compression functionality
****************

**Deprecation:** The Compression funtionality in the oneDAL library are marked as deprecated. From 2024.0 release of the oneDAL library will no longer include support for Compression functioanlity.

**Reasons for deprecation:** The ongoing efforts to optimize oneDAL resources and focus strongly on the most widely used features. 

**Alternatives:** Use external compression mechanics using optimzied impltementation such as Intel IPP in to your application

ABI compatibility
****************

**Deprecation:** ABI compatibility would be broken as part of 2024.0 release of the oneDAL library. Librariy major version would be increased to 2 to requre relinking of existing aplications

**Reasons for deprecation:**  Clean up of deprecated functionality, interfaces and symbols

**Alternatives:** Relink to newer version. Assume no ABI compatibility with migration to 2024 version

MacOS support
****************

**Deprecation:** MacOS support have been depricated for oneDAL and other oneAPI componens with 2023.x releases beeing last providing availability.

**Reasons for deprecation:**  No modern X86 MacOS based systems will be released.

**Alternatives:** Keep using 2023.x version on MacOS
