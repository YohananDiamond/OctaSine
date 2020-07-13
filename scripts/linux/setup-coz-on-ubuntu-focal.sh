#!/bin/sh
# prepare ubuntu focal system for coz profiling using process-bench
# (debian buster glibc version causes issues with coz)
#
# results for simd version mostly recommend improving speed of sleef sin
# function and simd intrinsics, maybe not very useful

apt-get update
apt-get upgrade -y
apt-get install screen

# Run code below in screen (especially coz installation since you will need to modify a file)

apt-get install git curl htop vim build-essential cmake -y
apt-get install clang docutils-common nodejs npm python3 python3-docutils pkg-config -y # coz dependencies

read "select nightly toolchain when prompted. press enter"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

cd "$HOME"
mkdir projects
cd projects

git pull https://github.com/aclements/libelfin.git
cd libelfin
make
make install
cd ..

git clone https://github.com/plasma-umass/coz.git
cd coz
read "comment out Makefile lines 14 and 18 seems to fix some issues. afterwards, press enter"
make
make install
cd ..

git pull https://github.com/greatest-ape/OctaSine.git
cd OctaSine
git checkout simd-avx
read "turn off LTO (really necessary now?), set benchmark iterations to suitable value (maybe 1_000_000), press enter"

./scripts/linux/bench-with-coz.sh

echo "upload 'profile.coz' to https://plasma-umass.org/coz/"