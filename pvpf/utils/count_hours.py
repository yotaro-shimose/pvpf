from datetime import datetime


def count_hours(start: datetime, end: datetime) -> int:
    dif = end - start
    assert dif.total_seconds() % 3600 == 0
    return int(dif.total_seconds() // 3600)
