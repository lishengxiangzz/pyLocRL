# pyLocRL

这个项目开发基于强化学习的定位智能体，涵盖目前现有的大部分定位体制。

- 侧向定位(Direction of Arrival, DOA)，根据辐射信号到达定位站的方位角进行定位。
- 到达时间差(Time of Arrival, TOA)，根据辐射信号到达定位站的传播时延，计算距离，三个定位站以距离为半径做圆进行交汇。
- 到达时间差(Time Difference of Arrival, TDOA)，又称为双曲线定位法。根据辐射信号到达各定位站点的时间差进行定位。
- 多普勒频率定位(FOA)，根据辐射信号到达定位站的载波频率和目标辐射信号的频率差进行定位。
- 相位差变化率定位(Phase Difference Shift)， 在定位站和与目标存在相对移动的情况下，将相对运动速度分解出切向速度，
  利用接收信号的相位差变化率，估算定位站与目标间的距离，再结合目标的方位角。

