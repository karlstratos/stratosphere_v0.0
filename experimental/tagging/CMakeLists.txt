CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

add_library(tagging STATIC util.cc util.h bilstm_tagger.cc bilstm_tagger.h)
target_link_libraries(tagging cnn)  # Must link my code to cnn!

ADD_EXECUTABLE(tagger tagger.cc)
target_link_libraries(tagger cnn tagging ${LIBS} pthread)

if(UNIX AND NOT APPLE)
  target_link_libraries(tagger rt)
endif()

if (WITH_CUDA_BACKEND)
  add_dependencies(tagger cnncuda)
  target_link_libraries(tagger cnncuda)
  CUDA_ADD_CUBLAS_TO_TARGET(tagger)
endif (WITH_CUDA_BACKEND)
