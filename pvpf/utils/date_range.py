from datetime import datetime, timedelta
from typing import Generator


def date_range(
    start: datetime, end: datetime, step: timedelta
) -> Generator[datetime, None, None]:
    cur = start
    while cur < end:
        yield cur
        cur += step
