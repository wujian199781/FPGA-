# FPGA-NESCers
第三届FPGA创新设计大赛《基于PYNQ-Z2的智能车流量检测》

本系统基于ETH的squeezenet做了改进和应用。如果想学习他们的源码，
请移步https://github.com/fpgasystems/spooNN/tree/master/halfsqueezenet



1.'jupyter notebook' 文件夹  为本工程在jupyter notebook上的操作代码。

2.'video' 是在 PYNQ-Z2板载文件 '\pynq\xilinx\pynq\lib\video\' 的路径下的改动代码，使用时应用此文件将原'video'文件替换。

3.'Vivado_HLS'为我们在执行vivado_hls高级综合时使用的代码。

4.vivado_project 包含了IP核 以及 生成block design 的 tcl 脚本。

5.'trainning' 包含了训练的python脚本，以及数据集等等
