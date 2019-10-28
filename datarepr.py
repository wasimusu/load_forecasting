import math
import numpy as np
import datetime as dt
from dateutil import parser


class DataRepresentation:
    @classmethod
    def repr_month(cls, month):
        angles = np.linspace(0, 2 * math.pi, 12)
        return math.sin(angles[month - 1])

    @classmethod
    def repr_day(cls, day):
        num_days = 31
        angles = np.linspace(0, 2 * math.pi, num_days)
        return math.sin(angles[day - 1])

    def get_week(cls, date):
        week = date.isoweekday()
        return week

    @classmethod
    def repr_hour(cls, hour):
        num_hours = 24
        angles = np.linspace(0, 2 * math.pi, num_hours)
        return math.sin(angles[hour - 1])


def repr_date(date):
    date = parser.parse(date)
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour

    # Sine representation
    month = DataRepresentation.repr_month(month)
    day = DataRepresentation.repr_day(day)
    hour = DataRepresentation.repr_hour(hour)

    return [year, month, day, hour]


if __name__ == '__main__':
    repr_date("2004-12-31 01:00:00")
