import math
import numpy as np
import datetime as dt


class ReprData:
    @classmethod
    def repr_month(cls, month):
        angles = np.linspace(0, 2 * math.pi, 12)
        return angles[month - 1]

    @classmethod
    def repr_day(cls, day):
        num_days = 30
        angles = np.linspace(0, 2 * math.pi, num_days)
        return angles[day - 1]

    def get_week(cls, date):
        week = date.isoweekday()

    @classmethod
    def repr_hour(cls, time):
        num_hours = 30
        hr = 2
        angles = np.linspace(0, 2 * math.pi, num_hours)
        return angles[hr - 1]


if __name__ == '__main__':
    date = dt.datetime(2019, 10, 16)
    print(date.isoweekday())
