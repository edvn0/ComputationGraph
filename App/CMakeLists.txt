cmake_minimum_required(VERSION 3.21)
project(App)

add_executable(App main.cpp)
target_link_libraries(App PRIVATE Core armadillo)

# If we are not on MSVC
if(NOT MSVC)
    # Enable C++17
    target_compile_options(Core PRIVATE -Wall -Wextra -Werror -Wpedantic)
else()
    # Aggressive msvc warnings
    target_compile_options(Core PRIVATE /W4 /WX)
endif()
