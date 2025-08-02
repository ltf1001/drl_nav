# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include;/usr/include/eigen3;/usr/local/include".split(';') if "${prefix}/include;/usr/include/eigen3;/usr/local/include" != "" else []
PROJECT_CATKIN_DEPENDS = "geometry_msgs;kdl_conversions;tf".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-ltf_conversions;/usr/local/lib/liborocos-kdl.so.1.5.1".split(';') if "-ltf_conversions;/usr/local/lib/liborocos-kdl.so.1.5.1" != "" else []
PROJECT_NAME = "tf_conversions"
PROJECT_SPACE_DIR = "/home/dell/drl/install_isolated"
PROJECT_VERSION = "1.13.2"
