def export_tab_file(freq_table, name, lat=0.0, long=0.0, height=0.0, dir_offset=0.0):
    lat = float(lat)
    long = float(long)
    speed_interval  = {interval.right - interval.left for interval in freq_table.index}
    if len(speed_interval)is not 1:
        import warnings
        warnings.warn("All speed bins not of equal lengths")
    speed_interval = speed_interval.pop()
    sectors = len(freq_table.columns)
    freq_sum = freq_table.sum(axis=0)
    freq_table.index = [interval.right for interval in freq_table.index]
    tab_string = str(name)+"\n "+"{:.2f}".format(lat)+" "+"{:.2f}".format(long)+" "+"{:.2f}".format(height)+"\n "
    tab_string += " ".join(str(percent) for percent in freq_sum.values)+"\n"
    for column in freq_table.columns:
        freq_table[column] = (freq_table[column] / sum(freq_table[column])) * 1000.0
    tab_string += freq_table.to_string(header=False, float_format='%.2f', na_rep=0.00)
    with open(str(name)+".tab", "w") as file:
        file.write(tab_string)
