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
        self.bath_rate = None
        self.dephasing_rate = None
        self.decay_rate = None
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

    def configure_chip(
            self, name: str, gate_time: float, decay_rate: float, dephasing_rate: float, bath_rate: float,
            T1: float, T2: float, dpc_error_rate: float, status: bool
    ):
        self.chipName = name
        self.gateTime = gate_time
        self.bath_rate = bath_rate
        self.decay_rate = decay_rate
        self.dephasing_rate = dephasing_rate
        self.T1 = T1
        self.T2 = T2
        self.dpc_errorRate = dpc_error_rate
        self.status = status

    def best(self):
        if self.queryTime is None:
            self.configure_chip(
                name='best',
                gate_time=30,
                bath_rate=0.,
                decay_rate=0.0,
                dephasing_rate=0.0,
                T1=2e11,
                T2=2e10,
                dpc_error_rate=11e-4,
                status=True
            )
        return self

    def medium(self):
        if self.queryTime is None:
            self.configure_chip(
                name='medium',
                gate_time=1,
                bath_rate=0.01,
                decay_rate=0.98,
                dephasing_rate=0.02,
                T1=2e11,
                T2=2e10,
                dpc_error_rate=5e-2,
                status=True
            )
        return self

    def worst(self):
        if self.queryTime is None:
            self.configure_chip(
                name='worst',
                gate_time=30,
                bath_rate=0.,
                decay_rate=0.0,
                dephasing_rate=0.0,
                T1=2e2,
                T2=2e1,
                dpc_error_rate=11e-2,
                status=True
            )
        return self

    def show_property(self):
        """
        Display the chip properties.
        """
        print(f"The chip name is: {self.chipName}")
        print(f"The gate time is: {self.gateTime} {self.timeUnit}")
        print(f"The T1 time is: {self.T1} {self.timeUnit}")
        print(f"The T2 time is: {self.T2} {self.timeUnit}")
        print(f"The depolarization error rate is: {self.dpc_errorRate}")
        print(f"The status of the chip is: {self.status}")
