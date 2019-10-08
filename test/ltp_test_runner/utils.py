def tabulate(values, columns_widths=None):
    if columns_widths is None:
        columns_widths = []
    cols = columns_widths + [15] * (len(values) - len(columns_widths))
    row_format = "".join(map(lambda n: "{%s:<%s}" % n, enumerate(cols)))
    return row_format.format(*values)
