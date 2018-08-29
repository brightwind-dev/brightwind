def _range_0_to_360(dir):
    if dir < 0:
        return dir+360
    elif dir > 360:
        return dir % 360
    else:
        return dir