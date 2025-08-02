# CMake generated Testfile for 
# Source directory: /home/dell/drl/src/geometry2/tf2_bullet
# Build directory: /home/dell/drl/build_isolated/tf2_bullet
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_tf2_bullet_gtest_test_bullet "/home/dell/drl/build_isolated/tf2_bullet/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/dell/drl/build_isolated/tf2_bullet/test_results/tf2_bullet/gtest-test_bullet.xml" "--return-code" "/home/dell/drl/devel_isolated/tf2_bullet/lib/tf2_bullet/test_bullet --gtest_output=xml:/home/dell/drl/build_isolated/tf2_bullet/test_results/tf2_bullet/gtest-test_bullet.xml")
set_tests_properties(_ctest_tf2_bullet_gtest_test_bullet PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;98;catkin_run_tests_target;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;37;_catkin_add_google_test;/home/dell/drl/src/geometry2/tf2_bullet/CMakeLists.txt;18;catkin_add_gtest;/home/dell/drl/src/geometry2/tf2_bullet/CMakeLists.txt;0;")
subdirs("gtest")
