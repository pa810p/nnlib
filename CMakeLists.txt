cmake_minimum_required(VERSION 3.31)
project(nnlib VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

option(NNLIB_ENABLE_COVERAGE "Instrument the project and generate an lcov report" OFF)

if(NNLIB_ENABLE_COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "NNLIB_ENABLE_COVERAGE currently requires GCC and gcov")
    endif()

    find_program(LCOV_EXECUTABLE lcov REQUIRED)
    find_program(GENHTML_EXECUTABLE genhtml REQUIRED)

endif()

set(NNLIB_SOURCES
    src/activation.cpp
    src/layer.cpp
    src/neuralnet.cpp
    src/neuron.cpp
)

include(FetchContent)
FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG v9.0.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

add_library(nnlib STATIC ${NNLIB_SOURCES})
add_library(nnlib_shared SHARED ${NNLIB_SOURCES})
set_target_properties(nnlib nnlib_shared PROPERTIES OUTPUT_NAME nn)

foreach(target IN ITEMS nnlib nnlib_shared)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_options(${target} PRIVATE -Wall)
    target_link_libraries(${target} PRIVATE glaze::glaze)
endforeach()

if(NNLIB_ENABLE_COVERAGE)
    target_compile_options(nnlib PRIVATE --coverage -O0 -g)
endif()

add_executable(demo app/main.cpp app/logger.cpp)
target_include_directories(demo PRIVATE app)
target_link_libraries(demo PRIVATE nnlib glaze::glaze)

if(NNLIB_ENABLE_COVERAGE)
    target_link_options(demo PRIVATE --coverage)
endif()

if(BUILD_TESTING)
    find_package(GTest REQUIRED)

    include(GoogleTest)

    function(add_nnlib_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE nnlib GTest::gtest_main)
        if(NNLIB_ENABLE_COVERAGE)
            target_link_options(${target} PRIVATE --coverage)
        endif()
        gtest_discover_tests(
            ${target}
            TEST_PREFIX "${target}."
            DISCOVERY_MODE PRE_TEST
        )
        set(NNLIB_TEST_TARGETS ${NNLIB_TEST_TARGETS} ${target} PARENT_SCOPE)
    endfunction()

    set(NNLIB_TEST_TARGETS)
    add_nnlib_test(test_activation tests/test_activation.cpp)
    add_nnlib_test(test_neuron tests/test_neuron.cpp)
    add_nnlib_test(test_layer tests/test_layer.cpp)
    add_nnlib_test(test_neuralnet tests/test_neuralnet.cpp)
    add_nnlib_test(test_logger tests/test_logger.cpp)
    target_include_directories(test_logger PRIVATE app)
    target_sources(test_logger PRIVATE app/logger.cpp)
    add_nnlib_test(test_neuralnet_scenarios tests/test_neuralnet_scenarios.cpp)
    add_nnlib_test(test_persistence tests/test_persistence.cpp)
    target_include_directories(test_persistence PRIVATE app)
    target_compile_definitions(test_persistence PRIVATE
        NNLIB_TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/resources"
    )
    target_link_libraries(test_persistence PRIVATE glaze::glaze)

    if(NNLIB_ENABLE_COVERAGE)
        add_custom_target(coverage
            COMMAND ${LCOV_EXECUTABLE} --directory ${CMAKE_BINARY_DIR} --zerocounters
            COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure
            COMMAND ${LCOV_EXECUTABLE}
                --directory ${CMAKE_BINARY_DIR}
                --capture
                --rc geninfo_unexecuted_blocks=1
                --output-file ${CMAKE_BINARY_DIR}/coverage.info
            COMMAND ${LCOV_EXECUTABLE}
                --extract ${CMAKE_BINARY_DIR}/coverage.info "${CMAKE_SOURCE_DIR}/src/*.cpp"
                --output-file ${CMAKE_BINARY_DIR}/coverage_filtered.info
            COMMAND ${LCOV_EXECUTABLE} --summary ${CMAKE_BINARY_DIR}/coverage_filtered.info
            COMMAND ${GENHTML_EXECUTABLE}
                ${CMAKE_BINARY_DIR}/coverage_filtered.info
                --output-directory ${CMAKE_BINARY_DIR}/coverage_html
            DEPENDS ${NNLIB_TEST_TARGETS}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Run tests and generate the HTML coverage report"
            VERBATIM
        )
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS nnlib nnlib_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES
    include/nnl/activation.h
    include/nnl/layer.h
    include/nnl/neuralnet.h
    include/nnl/neuron.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nnl
)
