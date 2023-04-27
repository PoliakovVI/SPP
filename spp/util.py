class SPPException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = ""

    def __str__(self):
        return f"[SPP ERROR]: {self.message}"


def parse_period(period):
    """
    period: X[y, mo, d]; X - number of units, [] - unit specificator
                E.g. 10d - 10 days period

    Return: [years, months, days]
    """
    # year month day
    ymd_list = [0, 0, 0]

    if period[-1] == "y":
        ymd_list[0] = int(period[:-1])
    elif period[-2:] == "mo":
        ymd_list[1] = int(period[:-2])
    elif period[-1] == "d":
        ymd_list[2] = int(period[:-1])
    else:
        raise SPPException(f"Period {period} has an unknown unit specificator")

    return ymd_list
