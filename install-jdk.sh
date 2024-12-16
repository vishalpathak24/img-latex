#/bin/bash
wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
tar -xvf openjdk-11+28_linux-x64_bin.tar.gz
export PATH=$PWD/jdk-11/bin:$PATH
java -version

