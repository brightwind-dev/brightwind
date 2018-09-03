def export_tab_file(freq_tab, name, lat=0.0, long=0.0, height=0.0, dir_offset=0.0):
    """Export a WaSP tab file from get_freq_table() function"""
    freq_tab  = freq_tab.copy()
    lat = float(lat)
    long = float(long)
    speed_interval  = {interval.right - interval.left for interval in freq_tab.index}
    if len(speed_interval)is not 1:
        import warnings
        warnings.warn("All speed bins not of equal lengths")
    speed_interval = speed_interval.pop()
    sectors = len(freq_tab.columns)
    freq_sum = freq_tab.sum(axis=0)
    freq_tab.index = [interval.right for interval in freq_tab.index]

    tab_string = str(name)+"\n "+"{:.2f}".format(lat)+" "+"{:.2f}".format(long)+" "+"{:.2f}".format(height)+"\n "+\
                 "{:.2f}".format(sectors)+" "+"{:.2f}".format(speed_interval)+" "+"{:.2f}".format(dir_offset)+"\n "
    tab_string += " ".join("{:.2f}".format(percent) for percent in freq_sum.values)+"\n"
    for column in freq_tab.columns:
        freq_tab[column] = (freq_tab[column] / sum(freq_tab[column])) * 1000.0
    tab_string += freq_tab.to_string(header=False, float_format='%.2f', na_rep=0.00)
    with open(str(name)+".tab", "w") as file:
        file.write(tab_string)
