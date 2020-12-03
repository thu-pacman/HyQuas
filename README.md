1. 如果qubit数量太少，程序会炸（已经在程序里用assert判断）
2. 如果qubit >= 31，下标会报int（已经在程序里assert判断，下标尽量都使用qIndex，但不保证没有漏）
3. 需要保证所有的measure和getAmp在run之后，未在代码中检查，现在没有做schedule，所以即便不满足结果暂时也是对的。后期可以把所有measure聚集起来
4. hardcode了$1/\sqrt{2}$的值，不知是否有更优雅的写法
5. 把门的具体内容放在了constant memory里，但constant memory容量有限，如果门比现在的测例多，可能会放不下。不知道放global会不会变慢，后期可以强拆成多次调用，或者直接为每个门编译一个kernel
6. 交换门的顺序后，会因为浮点误差的原因导致结果出错吗
7. 对角门的schedule暂时没有将作用在同一个qubit上的操作聚在一起
8. 数据在global mem 和 share mem 里的交换怎么写会比较快。。
TODO:

unroll

对角阵不用通信

因为0的缘故，前期的很多操作不会影响到所有值

把对同一个qubit的操作fuse成一个矩阵乘

X Gate 可以不算

写多卡时只考虑了 global qubit 不重叠的情况