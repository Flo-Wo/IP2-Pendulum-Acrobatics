import pandas as pd


def results_to_latex(
    x_err: pd.DataFrame,
    rot_err: pd.DataFrame,
    comp_time: pd.DataFrame,
    conf_overview: pd.DataFrame,
    target_folder: str,
    filenames: dict = {
        "x": "x_error",
        "rot": "rot_error",
        "comp_time": "comp_time",
        "conf": "solver_config",
    },
    captions: dict = {"x": "", "rot": "", "comp_time": "", "conf": ""},
    caption_meta: dict = {},
    filename_meta: dict = {},
    column_format: dict = {
        "x": "cccccccccccccccccccc",
        "rot": "cccccccccccccccccccc",
        "comp_time": "cccccccccccccccccccc",
        "conf": "ccccc",
    },
    how_bold: dict = {"x": "min", "rot": "min", "comp_time": "min", "conf": None},
):
    get_filename = _filename_mapper(
        target_folder, filenames=filenames, filename_meta=filename_meta
    )
    # create the tex files
    x_err_content = _pandas_to_tex(
        x_err,
        3,
        caption=captions["x"] + _unique_caption(**caption_meta),
        path_to_tex=get_filename("x"),
        column_format=column_format["x"],
        how_bold=how_bold["x"],
        replace_col_prefix=True,
    )

    rot_err_content = _pandas_to_tex(
        rot_err,
        3,
        caption=captions["rot"] + _unique_caption(**caption_meta),
        path_to_tex=get_filename("rot"),
        column_format=column_format["rot"],
        how_bold=how_bold["rot"],
        replace_col_prefix=True,
    )
    comp_time_content = _pandas_to_tex(
        comp_time,
        3,
        caption=captions["comp_time"] + _unique_caption(**caption_meta),
        path_to_tex=get_filename("comp_time"),
        column_format=column_format["comp_time"],
        how_bold=how_bold["comp_time"],
        replace_col_prefix=True,
    )
    conf_table = _pandas_to_tex(
        conf_overview,
        0,
        caption=captions["conf"] + _unique_caption(include_error=False, **caption_meta),
        path_to_tex=get_filename("conf"),
        column_format=column_format["conf"],
        how_bold=how_bold["conf"],
        replace_col_prefix=True,
        format_index=True,
        format_radius_orientation=False,
    )


def _filename_mapper(target_folder: str, filenames: dict, filename_meta: dict):
    meta_str = _unique_filename(**filename_meta)
    print("meta-str:")
    print(meta_str)

    print("target-folder")
    print(target_folder)

    def get_filename(mode: str):
        return target_folder + "/" + filenames[mode] + meta_str + ".tex"

    return get_filename


def _unique_filename(**kwargs):
    meta_str = "_"
    for key, value in kwargs.items():
        meta_str += "{}_{}_".format(key, value)
    return meta_str[:-1]


def _unique_caption(
    factor_integration_time: int = 1,
    end_error: int = 300,
    include_error: bool = True,
    **kwargs,
):
    unique_caption = ""
    if include_error:
        unique_caption += "Average error is computed for the last {0} nodes, ".format(
            end_error
        ) + "i.e. for the last \\SI{{{0}}}{{\\s}} of the trajectory. ".format(
            end_error * 0.002
        )
    unique_caption += "We used a factor of {} for the integration time, ".format(
        factor_integration_time
    ) + "i.e. the integration constant is \\SI{{{0}}}{{\\s}} with a real-time control frequency of \\SI{{{1}}}{{\\hertz}}.".format(
        0.002 * factor_integration_time,
        int(500 / factor_integration_time),
    )
    if kwargs:
        unique_caption += " Using "
        for key, value in kwargs.items():
            unique_caption += "${} = {}$, ".format(key, value)
        unique_caption = unique_caption[:-2]
        unique_caption += "."
    unique_caption += (
        "\\\\Bold types of minimum values are considered experiment (i.e. row) wise."
    )
    return unique_caption


def _pandas_to_tex(
    df: pd.DataFrame,
    ignore_cols: int = 3,
    caption: str = "",
    path_to_tex: str = None,
    column_format: str = "lrrllllllllllll",
    how_bold: str = "min",
    replace_col_prefix: bool = True,
    format_index: bool = False,
    format_radius_orientation: bool = True,
):

    # 2 times transpose as we want row based coloring: cols -> rows -> cols
    df = _format_cols_bold(
        df.transpose(),
        ignore_cols=ignore_cols,
        how=how_bold,
    ).transpose()

    df.columns = [col.replace("_", "-") for col in df.columns]
    if "dir" in df.columns:
        df = df.rename(columns={"dir": "direction"})
        df = df.set_index("direction")
    df = df.reset_index()
    if format_index:
        df = df.rename(columns={"index": "penalty"})
        df["penalty"] = df["penalty"].str.replace(r"_", "\\_", regex=True)

    if replace_col_prefix:
        df = _replace_col_prefix(df)

    if format_radius_orientation:
        df["orientation"] = df["orientation"].map("${{{0:+}}}$".format)
        df["radius"] = df["radius"].map("{:1.2f}".format)

    content = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        column_format=column_format,
    )

    if path_to_tex is not None:
        _save_table(content, path=path_to_tex)
    return content


def _replace_col_prefix(df: pd.DataFrame):
    cols = df.columns
    rename = {col: _rename_col(col) for col in cols}
    return df.rename(columns=rename)


def _rename_col(col: str):
    return col.replace("x-err-", "").replace("rot-err-", "").replace("comp-time-", "")


def _format_cols_bold(df: pd.DataFrame, ignore_cols: int = 3, how: str = "min"):
    """how = None or how="min" or how="max" """
    cols_to_show_max = list(df.columns)
    if isinstance(cols_to_show_max[0], str):
        return df

    for col in cols_to_show_max:

        # df.iloc[ignore_cols:, col] = df[ignore_cols:][col].apply(
        df.iloc[ignore_cols:, col] = df.iloc[ignore_cols:, col].apply(
            # df[ignore_cols:][col].apply(
            # df.iloc[ignore_cols:, col] = df[ignore_cols:][col].apply(
            lambda data: _bold_extreme_values(
                data,
                extreme_val=(
                    df[ignore_cols:][col].min()
                    if how == "min"
                    else df[ignore_cols:][col].max()
                    if how == "max"
                    else None
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
    extreme_val: float = None,
):
    cell = _float_exponent_notation(data, precision_digits=precision_digits)
    if extreme_val is not None and data == extreme_val:
        cell = "\\mathbf{{{0}}}".format(cell)

    cell = "${}$".format(cell)
    return cell


def _save_table(content: str, path: str):
    with open(path, "w") as file:
        file.write(content)
