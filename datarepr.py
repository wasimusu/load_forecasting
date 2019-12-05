import math
import numpy as np
import holidays
from dateutil import parser
import datetime as dt

us_holidays = holidays.UnitedStates()


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
    strdate = date  # String representation of date
    date = parser.parse(date)
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    is_holiday = 1 if us_holidays.get(key=dt.datetime(year, month, day)) else 0
    dayofweek = date.isoweekday()
    is_weekday = 1 if dayofweek in [6, 7] else 0

    # Sine representation
    if type == 'Sine':
        func = math.sin

    # Cosine representation
    elif type == 'Cosine':
        func = math.cos

    elif type == 'Pair':
        sa, sb, sc, d, is_holiday = repr_date(strdate, 'Sine')
        ca, cb, cc, _, _ = repr_date(strdate, 'Cosine')
        return [sa, ca, sb, cb, sc, cc, d, is_holiday, dayofweek, is_weekday]

    elif type == "Plain":
        return [year, month, day, hour, is_holiday, dayofweek, is_weekday]

    else:
        raise ValueError("Supported representations : Sine, Cosine, Pair. Input : {}".format(type))

    month = DataRepresentation.repr_month(month, func)
    day = DataRepresentation.repr_day(day, func)
    hour = DataRepresentation.repr_hour(hour, func)

    return [year, month, day, hour, is_holiday, dayofweek, is_weekday]


if __name__ == '__main__':
    repr_date("2004-12-31 01:00:00")
