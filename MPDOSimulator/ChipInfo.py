"""
Author: weiguo_ma
Time: 04.27.2023
Contact: weiguo.m@iphy.ac.cn
"""
class ChipInformation:
    def __init__(self, query_time: str = None):
        self.queryTime = query_time
        self.status = None

        self.gateTime = None
        self.T1 = None
        self.T2 = None
        self.chipName = None
        self.dpc_errorRate = None

        self.timeUnit = 'ns'

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            raise AttributeError(f'Chip: {item} is not supported.')

    def beta4Test(self):
        self.chipName = 'beta4Test'
        if self.queryTime is None:
            self.gateTime = 30
            self.T1 = 2e11
            self.T2 = 2e10
            self.dpc_errorRate = 11e-4
            self.status = True
        return self

    def worst4Test(self):
        """Set chip information for the worst4Test scenario."""
        self.chipName = 'worst4Test'
        if self.queryTime is None:
            self.gateTime = 30
            self.T1 = 2e2
            self.T2 = 2e1
            self.dpc_errorRate = 11e-2
            self.status = True
        return self

    def show_property(self):
        print('The chip name is: {}'.format(self.chipName))
        print('The gate time is: {} {}'.format(self.gateTime, self.timeUnit))
        print('The T1 time is: {} {}'.format(self.T1, self.timeUnit))
        print('The T2 time is: {} {}'.format(self.T2, self.timeUnit))
        print('The depolarization error rate is: {}'.format(self.dpc_errorRate))
        print('The status of the chip is: {}'.format(self.status))
