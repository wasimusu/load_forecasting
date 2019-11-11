import math
import numpy as np
import datetime as dt
from dateutil import parser


class DataRepresentation:
    @classmethod
    def repr_month(cls, month, func):
        angles = np.linspace(0, 2 * math.pi, 12)
        return func(angles[month - 1])

    @classmethod
    def repr_day(cls, day, func):
        num_days = 31
        angles = np.linspace(0, 2 * math.pi, num_days)
        return func(angles[day - 1])

    def get_week(cls, date):
        week = date.isoweekday()
        return week

    @classmethod
    def repr_hour(cls, hour, func):
        num_hours = 24
        angles = np.linspace(0, 2 * math.pi, num_hours)
        return func(angles[hour - 1])


def repr_date(date, type='Pair'):
    date = parser.parse(date)
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour

    # Sine representation
    if type == 'Sine':
        func = math.sin

    # Cosine representation
    elif type == 'Cosine':
        func = math.cos

    elif type == 'Pair':
        sa, sb, sc, d = repr_date(date, 'Sine')
        ca, cb, cc, _ = repr_date(date, 'Cosine')
        return [sa, ca, sb, cb, sc, cc, d]

    elif type == "Plain":
        return [year, month, day, hour]

    else:
        raise ValueError("Supported representations : Sine, Cosine, Pair. Input : {}".format(type))

    month = DataRepresentation.repr_month(month, func)
    day = DataRepresentation.repr_day(day, func)
    hour = DataRepresentation.repr_hour(hour, func)

    return [year, month, day, hour]


if __name__ == '__main__':
    repr_date("2004-12-31 01:00:00")
