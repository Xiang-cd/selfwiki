# wdt

不知道你有没有在自己的机器和服务器之间拷贝文件。

不知道你有没有在机器之间拷贝大的文件。

如果有, 不知道你有没有觉得scp很慢。

如果你在拷贝TB级别的文件, 是不是恨不得跑到机房用硬盘拷贝。

 那可以考虑用wdt来拷贝, 不说了, 直接上代码。



## 在mac上安装(手动)

```shell
# clone 仓库
git clone git@github.com:facebook/wdt.git
# 安装依赖
brew install cmake
brew install glog gflags boost
brew install double-conversion
brew install openssl
brew install fmt
git clone git@github.com:facebook/folly.git
git clone git@github.com:facebook/wdt.git
总之我在mac m1上没有安装成功, 链接错误,怪了。
cmake .. -DWDT_USE_SYSTEM_FOLLY=1 -DBUILD_TESTING=off  -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@1.1 -DBUILD_SHARED_LIBS=off \
      -DDOUBLECONV_INCLUDE_DIR="$HOME/include" \
      -DDOUBLECONV_LIBRARY="$HOME/lib/libdouble-conversion.dylib" && make -j

```



## linux

```shell
mkdir wdt_root && cd wdt_root
git clone https://github.com/facebook/folly.git
git clone git@github.com:facebook/wdt.git
sudo apt-get install libgoogle-glog-dev libboost-system-dev \
libdouble-conversion-dev libjemalloc-dev libfmt-dev libgtest-dev 
# 如果找不到各个库的源代码位置，就需要手动在cmake.txt 设置一下，或者百度后设置cmake的宏
makdir wdt-linux && cd wdt-linux
# 可能还需要注意glog 和 gflag的安装, 详见参考
# 中途编译出错不影响wdt应用程序的编译成功, 很多是一些测试程序编译错误了
cmake ../wdt
```



## 命令

一下是基本能work的命令，但是并没有perform加密手法，如果传输敏感数据还是需要再考虑一下

```shell
# 接收方
wdt -directory ~/Download -start_port 22356 --encryption_type=none --transfer_id=1234
# 发送方
 wdt -directory ./ --ipv4 -destination=接收方的ipv4地址 --encryption_type=none --transfer_id=1234
```

