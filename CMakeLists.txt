# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
# Licensed under the MIT license. See the license file LICENSE.                

project(DpMM)
cmake_minimum_required(VERSION 2.4.6)
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

if(NOT USE_OPENMP)
	#removes openMP flags
	remove_cxx_release_flag("-fopenmp")
	remove_cxx_debug_flag("-fopenmp")
endif()

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
	set(BUILD_SHARED_LIBS ON)
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
  #/usr/include/eigen3
  $ENV{WORKSPACE_HOME}/3rdparty/include
  ${Boost_INCLUDE_DIR}
  /usr/include
)

link_directories(
  $ENV{WORKSPACE_HOME}/3rdparty/lib
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
  ${Boost_LIBRARY_DIRS}
)

cuda_add_executable(testSphere ./test/sphere.cpp
	./src/normal.cpp
	./src/sphere.cpp
	./src/sampler.cpp
  ./cuda/log_p_kernel.cu
  ./cuda/karcherMeans_kernel.cu
  ./cuda/sufficientStatistics_kernel.cu
  ./cuda/sphereMixturePdf_kernel.cu
  ./cuda/gmmPdf_kernel.cu
  ./cuda/choiceMult_kernel.cu
)

target_link_libraries(testSphere
${Boost_LIBRARIES}
  #boost_system
  #boost_random
  #boost_unit_test_framework
)


cuda_add_executable(dpDirMM ./src/dpDirMM.cpp
  ./src/distribution.cpp
  ./src/niw.cpp
  ./src/iw.cpp
  ./src/normal.cpp
  ./src/cat.cpp
  ./src/mult.cpp
  ./src/dir.cpp
  ./src/niwSphere.cpp
  ./src/niwSphereFull.cpp
#  ./src/niwTangent.cpp
  ./src/sphere.cpp
  ./src/sampler.cpp
  ./cuda/choiceMult_kernel.cu
)
target_link_libraries(dpDirMM
${Boost_LIBRARIES}
  #boost_system
  #boost_random
  #boost_program_options
)

cuda_add_executable(dpmmSampler ./src/dpmmSampler.cpp
  ./src/distribution.cpp
  ./src/cat.cpp
  ./src/mult.cpp
  ./src/niw.cpp
  ./src/iw.cpp
  ./src/dir.cpp
  ./src/normal.cpp
  ./src/niwSphere.cpp
  ./src/niwSphereFull.cpp
#  ./src/niwTangent.cpp
  ./src/sphere.cpp
  ./src/sampler.cpp
#  ./src/gpuMatrix.cpp
  ./cuda/choiceMult_kernel.cu
  ./cuda/sphereMixturePdf_kernel.cu
  ./cuda/karcherMeans_kernel.cu
  ./cuda/log_p_kernel.cu
  ./cuda/gmmPdf_kernel.cu
  ./cuda/sufficientStatistics_kernel.cu
)
target_link_libraries(dpmmSampler
${Boost_LIBRARIES}
#boost_system
  #boost_random
  #boost_program_options
)



add_executable(generateSphericalData ./src/generateSphericalData.cpp
  ./src/normal.cpp
  ./src/iw.cpp
  ./src/sphere.cpp
)
target_link_libraries(generateSphericalData
${Boost_LIBRARIES}
  #boost_system
  #boost_random
  #boost_program_options
)

cuda_add_executable(myNaiveBayes 
	./test/naiveBayes.cpp
	./src/dir.cpp
	./src/cat.cpp
	./src/mult.cpp
	./src/niw.cpp
	./src/iw.cpp
	./src/normal.cpp
	./src/niwSphere.cpp
	./src/sampler.cpp
	./src/sphere.cpp
  ./cuda/choiceMult_kernel.cu
  ./cuda/log_p_kernel.cu
  ./cuda/karcherMeans_kernel.cu
  ./cuda/gmmPdf_kernel.cu
  ./cuda/sufficientStatistics_kernel.cu
  ./cuda/sphereMixturePdf_kernel.cu
)
target_link_libraries(myNaiveBayes
	${Boost_LIBRARIES}
)


cuda_add_executable(dirMultiObsNaiveBayes 
	./src/dirMultiObsNaiveBayes.cpp
	./src/dir.cpp
	./src/cat.cpp
	./src/mult.cpp
	./src/niw.cpp
	./src/iw.cpp
	./src/normal.cpp
	./src/niwSphere.cpp
	./src/sampler.cpp
	./src/sphere.cpp
  ./cuda/choiceMult_kernel.cu
  ./cuda/log_p_kernel.cu
  ./cuda/karcherMeans_kernel.cu
  ./cuda/gmmPdf_kernel.cu
  ./cuda/sufficientStatistics_kernel.cu
  ./cuda/sphereMixturePdf_kernel.cu
)
target_link_libraries(dirMultiObsNaiveBayes
	${Boost_LIBRARIES}
)


if( COMPILE_TEST ) 
	cuda_add_executable(testDirMM ./test/dirMM.cpp
		./src/dir.cpp
		./src/cat.cpp
		./src/mult.cpp
		./src/niw.cpp
		./src/iw.cpp
		./src/normal.cpp
	  #  ./src/niwBaseMeasure.cpp
		./src/niwSphere.cpp
		./src/sampler.cpp
		./src/sphere.cpp
	  #  ./src/gpuMatrix.cpp
	  ./cuda/choiceMult_kernel.cu
	  ./cuda/log_p_kernel.cu
	  ./cuda/karcherMeans_kernel.cu
	  ./cuda/gmmPdf_kernel.cu
	  ./cuda/sufficientStatistics_kernel.cu
	  ./cuda/sphereMixturePdf_kernel.cu
	)
	target_link_libraries(testDirMM
	${Boost_LIBRARIES}
	  #boost_system
	  #boost_random
	  #boost_unit_test_framework
	)



	cuda_add_executable(testSampler ./test/sampler.cpp
	  ./src/sampler.cpp
	  ./src/gpuMatrix.cpp
	  ./cuda/choiceMult_kernel.cu
	)
	target_link_libraries(testSampler
	${Boost_LIBRARIES}
	  #boost_system
	  #boost_random
	  #boost_unit_test_framework
	)


	cuda_add_executable(testClData ./test/clData.cpp
	  ./src/gpuMatrix.cpp
	  ./src/normal.cpp
	  ./src/sphere.cpp
	  ./cuda/log_p_kernel.cu
	  ./cuda/sphereMixturePdf_kernel.cu
	  ./cuda/karcherMeans_kernel.cu
	  ./cuda/sufficientStatistics_kernel.cu
	  ./cuda/gmmPdf_kernel.cu
	)
	target_link_libraries(testClData
	${Boost_LIBRARIES}
	  #boost_system
	  #boost_random
	  #boost_unit_test_framework
	)

	cuda_add_executable(testSPKM ./test/spkm.cpp
	  ./src/sphere.cpp
	  ./src/normal.cpp
	  ./src/cat.cpp
	  ./src/mult.cpp
	  ./src/dir.cpp
	)
	target_link_libraries(testSPKM
	${Boost_LIBRARIES}
	  #boost_system
	  #boost_random
	  #boost_unit_test_framework
	)

	cuda_add_executable(testDpSubclusterMM ./test/dpSubclusterMM.cpp
		./src/cat.cpp
		./src/mult.cpp
		./src/dir.cpp
		./src/niw.cpp
		./src/iw.cpp
		./src/normal.cpp
		./src/niwSphere.cpp
		./src/niwSphereFull.cpp
#		./src/niwTangent.cpp
		./src/sphere.cpp
	  ./src/sampler.cpp
	  ./src/gpuMatrix.cpp
	  ./cuda/choiceMult_kernel.cu
	)
	target_link_libraries(testDpSubclusterMM
	${Boost_LIBRARIES}
	  #boost_system
	  #boost_random
	  #boost_unit_test_framework
	)

	cuda_add_executable(testNaiveBayes 
		./test/test_naiveBayes.cpp
		./src/dir.cpp
		./src/cat.cpp
		./src/mult.cpp
		./src/niw.cpp
		./src/iw.cpp
		./src/normal.cpp
		./src/niwSphere.cpp
		./src/sampler.cpp
		./src/sphere.cpp
	  ./cuda/choiceMult_kernel.cu
	  ./cuda/log_p_kernel.cu
	  ./cuda/karcherMeans_kernel.cu
	  ./cuda/gmmPdf_kernel.cu
	  ./cuda/sufficientStatistics_kernel.cu
	  ./cuda/sphereMixturePdf_kernel.cu
	)
	target_link_libraries(testNaiveBayes
		${Boost_LIBRARIES}
	)

endif() 



#add_executable(dpSubclusterGMM ./src/dpSubclusterGMM.cpp
  #./src/dpSubclustersMM.cpp
  #./include/dpmmSubclusters/cluster_sampledT.cpp
  #./include/dpmmSubclusters/normal.cpp
  #./include/dpmmSubclusters/niw_sampled.cpp
#)
#target_link_libraries(dpSubclusterGMM
  #boost_system
  #boost_random
  #boost_program_options
  #gsl gslcblas m rt
#)

#add_executable(dpSubclusterSphereGMM ./src/dpSubclusterSphereGMM.cpp
#  ./src/dpSubclustersSphereMM.cpp
#  ./include/dpmmSubclusters/cluster_sampledT.cpp
#  ./include/dpmmSubclusters/normal.cpp
#  ./src/niwSphere_sampled.cpp
#)
#target_link_libraries(dpSubclusterSphereGMM
#  boost_system
#  boost_random
#  boost_program_options
#  gsl gslcblas m rt
#)

#cuda_add_executable(realtimeMF_openni
#  ./src/realtimeMF_openni.cpp
#  ./src/cv_helpers.cpp
#  ./src/pcl_helpers.cpp
#  ./src/realtimeMF.cpp
#  ./src/optimizationSO3.cpp
#  ./src/optimizationSO3_approx.cu
#  ./src/convolutionSeparable.cu
#  ./src/convolutionSeparable_small.cu
#  ./src/cuda_pcl_helpers.cu
#  ./src/optimizationSO3.cu)
#target_link_libraries (realtimeMF_openni
#  boost_system boost_thread
#  pcl_common pcl_io pcl_visualization pcl_filters
#  OpenNI XnVNite
#  vtkCommon vtkFiltering vtkRendering vtkGraphics
#  opencv_core opencv_highgui)


#add_executable(testDist ./test/distributions.cpp
    #./src/dir.cpp
  #./src/cat.cpp
  #./src/mult.cpp
    #./src/niw.cpp
    #./src/iw.cpp
    #./src/normal.cpp
  #./include/dpmmSubclusters/niw_sampled.cpp
  #./include/dpmmSubclusters/normal.cpp
#)
#target_link_libraries(testDist
  #boost_system
  #boost_random
  #boost_unit_test_framework
  #gsl gslcblas m rt
#)
