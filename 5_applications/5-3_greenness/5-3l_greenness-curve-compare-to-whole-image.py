
# ==========================================================
# GREENNESS CURVE
#
# Compares:
#   1. Deciduous-class GCC
#   2. Whole-image GCC
#
# For:
#   - SE-NM
#   - BE-LE
#
# Solid line  = deciduous class
# Dashed line = whole image
# ==========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# ==========================================================
# SETTINGS
# ==========================================================

DATASETS = [

    {
        "name": "SE-NM",

        # Class-level greenness CSV
        "class_csv":
            r"path-to-class-greenness-csv-1.csv",

        # Whole-image greenness CSV
        r"path-to-whole-greenness-1.csv":
            

        "colour": "#0000a2",
    },

    {
        "name": "BE-LE",

        # Class-level greenness CSV
        "class_csv":
            r"path-to-class-greenness-csv-2.csv",

        # Whole-image greenness CSV
        "whole_csv":
            r"path-to-whole-greenness-2.csv",

        "colour": "#bc272d",
    },
]

MANUAL_BREAKPOINTS = {

    "SE-NM": {
        "i": 152,
        "ii": 243,
        "iii": 299,
    },

    "BE-LE": {
        # Add BE-LE breakpoints here if wanted
        # "i": 120,
        # "ii": 200,
        # "iii": 300,
    },
}


SHOW_BREAKPOINTS = True
SHOW_BREAKPOINT_LINES = True

# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_DIR = Path(
    r"path-to-output-directory"
)


# ==========================================================
# CLASS TO PLOT
# ==========================================================

SELECTED_CLASS = (
    "class_13_broadleaf_deciduous"
)


# ==========================================================
# COLUMN NAMES
# ==========================================================

DATE_COL = "date"

CLASS_COL = "class_folder"

VALUE_COL = "GCC"


# ==========================================================
# PLOT MODE
# ==========================================================

# "annual_cycle"
# or
# "continuous"

PLOT_MODE = "annual_cycle"


# ==========================================================
# ERROR
# ==========================================================

ERROR_TYPE = "std"

# Options:
# "std"
# "sem"
# "none"


# ==========================================================
# SMOOTHING
# ==========================================================

APPLY_SAVGOL = True

SAVGOL_WINDOW = 75

SAVGOL_POLYORDER = 3


# ==========================================================
# FIGURE SETTINGS
# ==========================================================

FIG_WIDTH = 12
FIG_HEIGHT = 9

DPI = 300


FONT_SIZE = 18

TITLE_SIZE = 18

AXIS_LABEL_SIZE = 18

TICK_SIZE = 18

LEGEND_SIZE = 16


LINE_WIDTH = 2.5

ERROR_ALPHA = 0.10


TITLE = (
    "Deciduous and whole-image mean annual greenness"
)

Y_LABEL = (
    "Greenness Chromatic Coordinate (GCC)"
)


# ==========================================================
# LINE STYLES
# ==========================================================

CLASS_LINESTYLE = "-"

WHOLE_LINESTYLE = "--"


# ==========================================================
# DISPLAY OPTIONS
# ==========================================================

SHOW_POINTS = False

SHOW_ERROR_BARS = True

SHOW_SMOOTHED_LINE = True

SHOW_TITLE = False


# ==========================================================
# SAVE OPTIONS
# ==========================================================

SAVE_PNG = True

SAVE_PDF = True

SHOW_FIGURE = True


# ==========================================================
# FONT SETTINGS
# ==========================================================

plt.rcParams.update({

    "font.family": "Arial",

    "font.size":
        FONT_SIZE,

    "axes.titlesize":
        TITLE_SIZE,

    "axes.labelsize":
        AXIS_LABEL_SIZE,

    "xtick.labelsize":
        TICK_SIZE,

    "ytick.labelsize":
        TICK_SIZE,

    "legend.fontsize":
        LEGEND_SIZE,
})


# ==========================================================
# FUNCTIONS
# ==========================================================

def ensure_odd_window(window):

    if window % 2 == 0:

        window += 1

    return window


# ----------------------------------------------------------
# SAVITZKY-GOLAY
# ----------------------------------------------------------

def safe_savgol(
    y,
    window,
    polyorder
):

    y = np.asarray(
        y,
        dtype=float
    )


    if len(y) < 3:

        return y


    window = ensure_odd_window(
        window
    )


    if window > len(y):

        if len(y) % 2 == 1:

            window = len(y)

        else:

            window = (
                len(y) - 1
            )


    if (
        window <= polyorder
        or
        window < 3
    ):

        return y


    return savgol_filter(

        y,

        window_length=window,

        polyorder=polyorder
    )


# ==========================================================
# LOAD CLASS DATA
# ==========================================================

def load_class_data(dataset):

    df = pd.read_csv(
        dataset["class_csv"]
    )


    required_cols = [

        DATE_COL,

        CLASS_COL,

        VALUE_COL,
    ]


    for col in required_cols:

        if col not in df.columns:

            raise ValueError(

                f"Missing column "
                f"'{col}' in:\n"
                f"{dataset['class_csv']}"
            )


    # Convert date

    df[DATE_COL] = pd.to_datetime(

        df[DATE_COL],

        errors="coerce"
    )


    # Convert GCC

    df[VALUE_COL] = pd.to_numeric(

        df[VALUE_COL],

        errors="coerce"
    )


    # Remove missing values

    df = df.dropna(

        subset=[

            DATE_COL,

            VALUE_COL,

            CLASS_COL,
        ]
    )


    # Select deciduous only

    df = df[

        df[CLASS_COL]
        ==
        SELECTED_CLASS

    ].copy()


    # Add dataset name

    df["dataset"] = (
        dataset["name"]
    )


    # Identify source

    df["source"] = (
        "Deciduous"
    )


    return df


# ==========================================================
# LOAD WHOLE-IMAGE DATA
# ==========================================================

def load_whole_data(dataset):

    df = pd.read_csv(
        dataset["whole_csv"]
    )


    required_cols = [

        DATE_COL,

        VALUE_COL,
    ]


    for col in required_cols:

        if col not in df.columns:

            raise ValueError(

                f"Missing column "
                f"'{col}' in:\n"
                f"{dataset['whole_csv']}"
            )


    # Convert date

    df[DATE_COL] = pd.to_datetime(

        df[DATE_COL],

        errors="coerce"
    )


    # Convert GCC

    df[VALUE_COL] = pd.to_numeric(

        df[VALUE_COL],

        errors="coerce"
    )


    # Remove missing values

    df = df.dropna(

        subset=[

            DATE_COL,

            VALUE_COL,
        ]
    )


    # Add dataset name

    df["dataset"] = (
        dataset["name"]
    )


    # Identify source

    df["source"] = (
        "Whole image"
    )


    return df


# ==========================================================
# AGGREGATE DATA
# ==========================================================

def aggregate_data(df):


    # ------------------------------------------------------
    # CONTINUOUS
    # ------------------------------------------------------

    if PLOT_MODE == "continuous":

        group_cols = [

            "dataset",

            "source",

            DATE_COL,
        ]

        x_col = DATE_COL


    # ------------------------------------------------------
    # ANNUAL CYCLE
    # ------------------------------------------------------

    elif PLOT_MODE == "annual_cycle":

        df = df.copy()


        # Remove 29 February so leap years do not shift
        # March-December by one day.

        df = df[

            ~(

                (df[DATE_COL].dt.month == 2)

                &

                (df[DATE_COL].dt.day == 29)

            )

        ].copy()


        # Map all dates to a common non-leap year.

        reference_dates = pd.to_datetime({

            "year":
                np.full(
                    len(df),
                    2021
                ),

            "month":
                df[DATE_COL].dt.month,

            "day":
                df[DATE_COL].dt.day,
        })


        df["annual_day"] = (

            reference_dates.dt.dayofyear
        )


        group_cols = [

            "dataset",

            "source",

            "annual_day",
        ]


        x_col = "annual_day"


    else:

        raise ValueError(

            "PLOT_MODE must be "
            "'continuous' or "
            "'annual_cycle'"
        )


    # ------------------------------------------------------
    # CALCULATE DAILY STATISTICS
    # ------------------------------------------------------

    grouped = (

        df.groupby(
            group_cols
        )[VALUE_COL]

        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )

        .reset_index()
    )


    grouped["std"] = (

        grouped["std"]
        .fillna(0)
    )


    grouped["sem"] = (

        grouped["std"]

        /

        np.sqrt(
            grouped["count"]
        )
    )


    # ------------------------------------------------------
    # SELECT ERROR TYPE
    # ------------------------------------------------------

    if ERROR_TYPE == "std":

        grouped["error"] = (
            grouped["std"]
        )


    elif ERROR_TYPE == "sem":

        grouped["error"] = (
            grouped["sem"]
        )


    elif ERROR_TYPE == "none":

        grouped["error"] = 0


    else:

        raise ValueError(

            "ERROR_TYPE must be "
            "'std', 'sem', or 'none'"
        )


    return (
        grouped,
        x_col
    )


# ==========================================================
# MAIN
# ==========================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True
)


all_data = []


# ----------------------------------------------------------
# LOAD BOTH DATA TYPES FOR EACH DATASET
# ----------------------------------------------------------

for dataset in DATASETS:


    # Deciduous class

    class_df = load_class_data(
        dataset
    )


    print(
        f"{dataset['name']} "
        f"deciduous observations: "
        f"{len(class_df)}"
    )


    all_data.append(
        class_df[
            [
                DATE_COL,
                VALUE_COL,
                "dataset",
                "source"
            ]
        ]
    )


    # Whole image

    whole_df = load_whole_data(
        dataset
    )


    print(
        f"{dataset['name']} "
        f"whole-image observations: "
        f"{len(whole_df)}"
    )


    all_data.append(
        whole_df[
            [
                DATE_COL,
                VALUE_COL,
                "dataset",
                "source"
            ]
        ]
    )


# ==========================================================
# COMBINE
# ==========================================================

df_all = pd.concat(

    all_data,

    ignore_index=True
)


if df_all.empty:

    raise ValueError(
        "No greenness data found."
    )


# ==========================================================
# AGGREGATE
# ==========================================================

summary, x_col = aggregate_data(
    df_all
)


# ==========================================================
# CREATE FIGURE
# ==========================================================

fig, ax = plt.subplots(

    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT
    )
)


# ==========================================================
# PLOT EACH DATASET
# ==========================================================

for dataset in DATASETS:


    dname = (
        dataset["name"]
    )


    colour = (
        dataset["colour"]
    )


    # ======================================================
    # PLOT DECIDUOUS + WHOLE IMAGE
    # ======================================================

    for source in [

        "Deciduous",

        "Whole image",

    ]:


        plot_df = summary[

            (
                summary["dataset"]
                ==
                dname
            )

            &

            (
                summary["source"]
                ==
                source
            )

        ].copy()


        plot_df = plot_df.sort_values(
            x_col
        )


        if plot_df.empty:

            continue


        x = (
            plot_df[x_col]
            .to_numpy()
        )


        y = (
            plot_df["mean"]
            .to_numpy()
        )


        err = (

            plot_df["error"]

            .fillna(0)

            .to_numpy()
        )


        # --------------------------------------------------
        # LINE STYLE
        # --------------------------------------------------

        if source == "Deciduous":
            linestyle = CLASS_LINESTYLE
            line_alpha = 1.0
        else:
            linestyle = WHOLE_LINESTYLE
            line_alpha = 0.5


        # --------------------------------------------------
        # SMOOTH
        # --------------------------------------------------

        if APPLY_SAVGOL:

            y_smooth = safe_savgol(

                y,

                SAVGOL_WINDOW,

                SAVGOL_POLYORDER
            )

        else:

            y_smooth = y


        # --------------------------------------------------
        # ERROR BAND
        # --------------------------------------------------

        if (
            SHOW_ERROR_BARS
            and
            ERROR_TYPE != "none"
        ):

            ax.fill_between(

                x,

                y - err,

                y + err,

                color=colour,

                alpha=ERROR_ALPHA,

                linewidth=0,
            )


        # --------------------------------------------------
        # POINTS
        # --------------------------------------------------

        if SHOW_POINTS:

            ax.plot(

                x,

                y,

                marker="o",

                markersize=3,

                linestyle="none",

                color=colour,

                alpha=0.7,
            )

        # ======================================================
        # MANUAL BREAKPOINTS
        # Only apply to the deciduous-class curve
        # ======================================================

        if (
                source == "Deciduous"
                and dname in MANUAL_BREAKPOINTS
        ):

            for breakpoint_name, doy in MANUAL_BREAKPOINTS[dname].items():

                # Get GCC value from the smoothed deciduous curve
                y_break = np.interp(
                    doy,
                    x,
                    y_smooth
                )

                # ----------------------------------------------
                # Vertical breakpoint line
                # ----------------------------------------------

                if SHOW_BREAKPOINT_LINES:
                    ax.axvline(
                        x=doy,
                        color=colour,
                        linestyle=":",
                        linewidth=1.4,
                        alpha=0.55,
                        zorder=4,
                    )

                # ----------------------------------------------
                # Breakpoint dot + label
                # ----------------------------------------------

                if SHOW_BREAKPOINTS:
                    ax.plot(
                        doy,
                        y_break,
                        marker="o",
                        markersize=6,
                        color=colour,
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        zorder=6,
                    )

                    ax.text(
                        doy,
                        y_break + 0.003,
                        breakpoint_name,
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        fontweight="bold",
                        color=colour,
                        zorder=7,
                    )

        # --------------------------------------------------
        # LINE
        # --------------------------------------------------

        if SHOW_SMOOTHED_LINE:

            y_plot = y_smooth

        else:

            y_plot = y


        ax.plot(

            x,

            y_plot,

            color=colour,

            linestyle=linestyle,

            linewidth=LINE_WIDTH,

            alpha=line_alpha,

            label=(
                f"{dname} – {source}"
            ),
        )


# ==========================================================
# AXIS LABELS
# ==========================================================

if SHOW_TITLE:

    ax.set_title(

        TITLE,

        fontsize=TITLE_SIZE
    )


ax.set_ylabel(

    Y_LABEL,

    fontsize=AXIS_LABEL_SIZE
)


# ==========================================================
# X AXIS
# ==========================================================

if PLOT_MODE == "continuous":


    ax.set_xlabel(

        "Date",

        fontsize=AXIS_LABEL_SIZE
    )


else:


    ax.set_xlabel(

        "Month",

        fontsize=AXIS_LABEL_SIZE
    )


    month_starts = [

        1,

        32,

        60,

        91,

        121,

        152,

        182,

        213,

        244,

        274,

        305,

        335,
    ]


    month_labels = [

        "Jan",

        "Feb",

        "Mar",

        "Apr",

        "May",

        "Jun",

        "Jul",

        "Aug",

        "Sep",

        "Oct",

        "Nov",

        "Dec",
    ]


    ax.set_xticks(
        month_starts
    )


    ax.set_xticklabels(
        month_labels
    )


    ax.set_xlim(
        1,
        365
    )


# ==========================================================
# FORMATTING
# ==========================================================

ax.tick_params(

    axis="both",

    labelsize=TICK_SIZE
)


ax.grid(

    True,

    alpha=0.3
)


ax.legend(

    fontsize=LEGEND_SIZE,

    loc="upper right"
)


fig.tight_layout()


# ==========================================================
# SAVE
# ==========================================================

output_base = (

    OUTPUT_DIR

    /

    f"deciduous_vs_whole_image_greenness_{PLOT_MODE}"
)


if SAVE_PNG:

    fig.savefig(

        f"{output_base}.png",

        dpi=DPI,

        bbox_inches="tight"
    )


if SAVE_PDF:

    fig.savefig(

        f"{output_base}.pdf",

        bbox_inches="tight"
    )


# ==========================================================
# SHOW
# ==========================================================

if SHOW_FIGURE:

    plt.show()

else:

    plt.close(fig)


print("\nDone.")

print(
    f"Figures saved to:\n"
    f"{OUTPUT_DIR}"
)
