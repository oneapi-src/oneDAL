sudo apt-get update
sudo apt-get install build-essential gcc gfortran cmake -y
git clone https://github.com/oneapi-src/oneTBB.git
CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)

rm -rf __deps/tbb
pushd oneTBB
git checkout v2021.11.0
mkdir build
pushd build
    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb  .. 
    make -j${CoreCount} 
    make install
popd
popd
rm -rf oneTBB

pushd __deps/tbb
    mkdir -p lnx
    mv lib/ lnx/
    mv include/ lnx/ 
    pushd lnx
        mkdir -p lib/arm/gcc4.8
        mv lib/libtbb* lib/arm/gcc4.8
    popd
popd 
