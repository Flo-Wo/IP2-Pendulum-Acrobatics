import pandas as pd

# inspired by:
# - https://blog.martisak.se/2021/04/10/publication_ready_tables/
# - https://flopska.com/highlighting-pandas-to_latex-output-in-bold-face-for-extreme-values.html

# formatting taken from:
# https://stackoverflow.com/questions/41157879/python-pandas-how-to-format-big-numbers-in-powers-of-ten-in-latex


def python_to_latex(
    df: pd.DataFrame,
    ignore_cols: int = 3,
    caption: str = "",
    path_to_tex: str = None,
    column_format: str = "lrrllllllllllll",
    how_bold: str = "min",
    show_index: bool = False,
):

    # 2 times transpose as we want row based coloring: cols -> rows -> cols
    df = _format_cols_bold(
        df.transpose(), ignore_cols=ignore_cols, how=how_bold
    ).transpose()

    df.columns = [col.replace("_", "-") for col in df.columns]
    df = df.set_index("direction").transpose()
    print(df)
    content = df.to_latex(
        index=show_index,
        escape=False,
        caption=caption,
        column_format=column_format,
        # header="",
    )
    if path_to_tex is not None:
        _save_table(content, path=path_to_tex)
    return content


def _format_cols_bold(df: pd.DataFrame, ignore_cols: int = 3, how: str = "min"):
    """how = None or how="min" or how="max" """
    if how is None:
        return df

    cols_to_show_max = df.columns

    for col in cols_to_show_max:
        df.loc[ignore_cols:, col] = df[ignore_cols:][col].apply(
            lambda data: _bold_extreme_values(
                data,
                extreme_val=(
                    df[ignore_cols:][col].min()
                    if how == "min"
                    else df[ignore_cols:][col].max()
                ),
            )
        )
    return df


def _float_exponent_notation(float_number, precision_digits, format_type="e"):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with `precision_digits` digits of
    mantissa precision, printing a normal decimal if an
    exponent isn't necessary.
    """
    e_float = "{0:.{1:d}{2}}".format(float_number, precision_digits, format_type)
    if "e" not in e_float:
        return "{}".format(e_float)
    mantissa, exponent = e_float.split("e")
    # TODO: check whether we want always the sign in the exponent
    # cleaned_exponent = exponent.strip("+")
    return "{0} \\cdot 10^{{{1}}}".format(mantissa, exponent)


def _bold_extreme_values(
    data,
    precision_digits: int = 3,
    extreme_val: float = 0,
):
    cell = _float_exponent_notation(data, precision_digits=precision_digits)
    if data == extreme_val:
        cell = "\\mathbf{{{0}}}".format(cell)

    cell = "${}$".format(cell)
    return cell


def _save_table(content: str, path: str):
    with open(path, "w") as file:
        file.write(content)
