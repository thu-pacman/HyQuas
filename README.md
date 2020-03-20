1. 如果qubit数量太少，程序会炸（已经在程序里用assert判断）
2. 如果qubit >= 31，下标会报int（已经在程序里assert判断，下标尽量都使用qIndex，但不保证没有漏）
3. 需要保证所有的measure和getAmp在run之后，未在代码中检查，现在没有做schedule，所以即便不满足结果暂时也是对的。后期可以把所有measure聚集起来
4. hardcode了$1/\sqrt{2}$的值，不知是否有更优雅的写法
5. 把门的具体内容放在了constant memory里，但constant memory容量有限，如果门比现在的测例多，可能会放不下。不知道放global会不会变慢，后期可以强拆成多次调用，或者直接为每个门编译一个kernel
TODO:
unroll, scheduler