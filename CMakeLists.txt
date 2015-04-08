# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.

project(DpMM)
cmake_minimum_required(VERSION 2.4.6)

set(POD_NAME DpMM)
include(cmake/pods.cmake)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#set( CMAKE_VERBOSE_MAKEFILE on )

set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
set(CMAKE_BUILD_TYPE Release) # Release; debug; ...

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(USE_OPENMP "Uses OpenMP (default true)" ON)
option(COMPILE_TEST "Specify whether to compile test programs (default true)" ON)


macro(remove_cxx_release_flag flag)
  string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endmacro()

macro(remove_cxx_debug_flag flag)
  string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endmacro()

macro(remove_cuda_flag flag)
  string(REPLACE "${flag}" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
endmacro()

# add flags depending on build type (these are the default flags, remove according to configuraiton (eg, windows configuration)
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG -DCUDA")
set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -g -Wall -Wno-deprecated  -DCUDA")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")

#by default build shared_libs
set(BUILD_SHARED_LIBS ON)

if(NOT USE_OPENMP)
	#removes openMP flags
	remove_cxx_release_flag("-fopenmp")
	remove_cxx_debug_flag("-fopenmp")
endif()

set(COMPILE_TEST ON)

if(WIN32)
	#windows configuration
	message("removing flags not available in MSVSC")
	remove_cxx_release_flag("-Wno-deprecated")
	remove_cxx_debug_flag("-Wno-deprecated")
	
	remove_cxx_release_flag("-finline-functions")
	remove_cxx_debug_flag("-finline-functions")
	
	remove_cxx_release_flag("-g")
	remove_cxx_debug_flag("-g")
	
	remove_cxx_release_flag("-finline-functions")
	remove_cxx_debug_flag("-finline-functions")
	
	remove_cxx_release_flag("-Wall")
	remove_cxx_debug_flag("-Wall")
	
	remove_cxx_release_flag("-fopenmp")
	remove_cxx_debug_flag("-fopenmp")

	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  /MP /MD /W0")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  /MDd /Zi /MP /Od")
	
	if( USE_OPENMP)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  /openmp")
	endif()
	
	#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-m64")
	#set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-machine:X64")
	#set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "-machine:X64")
	
	#remove_cuda_flag("-fopenmp")
	#remove_cuda_flag("-machine:X64")
	
	set(Boost_USE_STATIC_LIBS   ON)
	set(Boost_USE_STATIC_RUNTIME OFF)

	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
	#set(CUDA_SEPARABLE_COMPILATION OFF)
	set(BUILD_SHARED_LIBS OFF)
	set(CUDA_VERBOSE_BUILD ON)

	set(Boost_DEBUG 1)
	
	set(COMPILE_TEST OFF)
endif()

find_package (Eigen3 REQUIRED)

set(Boost_NO_SYSTEM_PATHS ON)
find_package(Boost COMPONENTS program_options random system unit_test_framework REQUIRED )

find_package (CUDA REQUIRED)

if(USE_OPENMP)
	find_package (OpenMP REQUIRED)
endif()

message("Eigen inc dir:" ${EIGEN3_INCLUDE_DIR})
message("CUDA toolkit dir:" ${CUDA_TOOLKIT_ROOT_DIR})
#message($ENV{BOOST_ROOT})
message("BOOST inc dir:" ${Boost_INCLUDE_DIR})
message("BOOST lib dir:" ${Boost_LIBRARY_DIRS})
message("BOOST libs:" ${Boost_LIBRARIES})

include_directories(
  ./include
  ./cuda
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
  ${Boost_INCLUDE_DIR}
  /usr/include
)

link_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
  ${Boost_LIBRARY_DIRS}
)

add_library(dpmm
  ./src/distribution.cpp
  ./src/cat.cpp
  ./src/mult.cpp
  ./src/niw.cpp
  ./src/iw.cpp
  ./src/dir.cpp
  ./src/normal.cpp
  ./src/niwSphere.cpp
  ./src/niwSphereFull.cpp
  ./src/sphere.cpp
)

cuda_add_library(dpmm_cuda
  ./src/sampler.cpp
  ./cuda/choiceMult_kernel.cu
  ./cuda/sphereMixturePdf_kernel.cu
  ./cuda/karcherMeans_kernel.cu
  ./cuda/log_p_kernel.cu
  ./cuda/gmmPdf_kernel.cu
  ./cuda/sufficientStatistics_kernel.cu
)

cuda_add_executable(testSphere ./test/sphere.cpp
)

target_link_libraries(testSphere
${Boost_LIBRARIES}
dpmm dpmm_cuda
)

cuda_add_executable(dpDirMM ./src/dpDirMM.cpp
)
target_link_libraries(dpDirMM
${Boost_LIBRARIES}
dpmm dpmm_cuda
)

cuda_add_executable(dpmmSampler ./src/dpmmSampler.cpp
)
target_link_libraries(dpmmSampler
  ${Boost_LIBRARIES}
  dpmm dpmm_cuda
)



add_executable(generateSphericalData ./src/generateSphericalData.cpp
)
target_link_libraries(generateSphericalData
  ${Boost_LIBRARIES}
  dpmm dpmm_cuda
  )

cuda_add_executable(myNaiveBayes
	./test/naiveBayes.cpp
)
target_link_libraries(myNaiveBayes
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
)


cuda_add_executable(dirMultiObsNaiveBayes
	./src/dirMultiObsNaiveBayes.cpp
)
target_link_libraries(dirMultiObsNaiveBayes
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
)


if( COMPILE_TEST )
	cuda_add_executable(testDirMM ./test/dirMM.cpp
	)
	target_link_libraries(testDirMM
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
	)

	cuda_add_executable(testCrpMM ./test/crpMM.cpp
	)
	target_link_libraries(testCrpMM
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
	)


	cuda_add_executable(testSampler ./test/sampler.cpp
	)
	target_link_libraries(testSampler
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
	)

	cuda_add_executable(testSPKM ./test/spkm.cpp
	)
	target_link_libraries(testSPKM
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
	)

	cuda_add_executable(testDpSubclusterMM ./test/dpSubclusterMM.cpp
	)
	target_link_libraries(testDpSubclusterMM
	${Boost_LIBRARIES}
  dpmm dpmm_cuda
	)

	cuda_add_executable(testNaiveBayes
		./test/test_naiveBayes.cpp
	)
	target_link_libraries(testNaiveBayes
		${Boost_LIBRARIES}
    dpmm dpmm_cuda
	)

  add_executable(testDist ./test/distributions.cpp
  )
  target_link_libraries(testDist
      ${Boost_LIBRARIES}
      dpmm dpmm_cuda
  )

endif()


INSTALL(TARGETS dpmm dpmmSampler
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  COMPONENT compLib
  )

pods_install_executables(dpmmSampler)
pods_install_libraries(dpmm)
pods_install_headers(
  ./include/dpMM/niw.hpp
  ./include/dpMM/unifSphere.hpp
  ./include/dpMM/kmeans.hpp
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  ./include/dpMM/
  DESTINATION dpMM)


#install(DIRECTORY include/ DESTINATION include/dpMM
#  FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY cuda/ DESTINATION include/dpMM
#  FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/ DESTINATION include/dpMM
#  FILES_MATCHING PATTERN "*.hpp")
