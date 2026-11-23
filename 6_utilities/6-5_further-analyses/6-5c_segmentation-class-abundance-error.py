
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# USER SETTINGS
# ============================================================

# Input confusion matrices
# These should contain RAW PIXEL COUNTS, not percentages.

SE_NM_FILE = Path(
    r"path-to-dataset-1-confusion-matrix.xlsx"
)

BE_LE_FILE = Path(
    r"path-to-dataset-2-confusion-matrix.xlsx"
)

# Excel sheet
SHEET_NAME = 0


# ------------------------------------------------------------
# Matrix format
# ------------------------------------------------------------

# Column containing the ground-truth class name
GT_CLASS_COLUMN = "ground_truth_class"

# Prediction columns are assumed to start with this prefix.
#
# Example:
# pred_Background
# pred_Sky
# pred_Conifer
PRED_PREFIX = "pred_"


# ------------------------------------------------------------
# Background
# ------------------------------------------------------------

BACKGROUND_CLASS = "gt-background"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"path-to-output-directory\abundance-error"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Font
# ------------------------------------------------------------

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# ============================================================
# LOAD CONFUSION MATRIX
# ============================================================

def load_confusion_matrix(path):

    if path.suffix.lower() in [".xlsx", ".xls"]:

        df = pd.read_excel(
            path,
            sheet_name=SHEET_NAME
        )

    elif path.suffix.lower() == ".csv":

        df = pd.read_csv(
            path
        )

    else:

        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    return df


# ============================================================
# ANALYSE ONE BIOME
# ============================================================

def analyse_confusion_matrix(
    df,
    biome_name
):

    # --------------------------------------------------------
    # Prediction columns
    # --------------------------------------------------------

    pred_columns = [
        column
        for column in df.columns
        if column.startswith(PRED_PREFIX)
    ]

    if len(pred_columns) == 0:

        raise ValueError(
            "No prediction columns found. "
            f"Expected columns beginning with '{PRED_PREFIX}'."
        )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    gt_classes = (
        df[GT_CLASS_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("gt_", "", regex=False)
        .tolist()
    )

    pred_classes = [
        column.strip()
        .lower()
        .replace(PRED_PREFIX.lower(), "")
        for column in pred_columns
    ]

    print(f"\n{biome_name}")
    print("=" * 60)

    print(
        f"GT classes: {len(gt_classes)}"
    )

    print(
        f"Prediction classes: {len(pred_classes)}"
    )

    # --------------------------------------------------------
    # Confusion matrix as array
    # --------------------------------------------------------

    matrix = (
        df[pred_columns]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Ground-truth totals
    # --------------------------------------------------------

    gt_totals = matrix.sum(
        axis=1
    )

    total_gt_pixels = gt_totals.sum()

    gt_abundance = (
        gt_totals /
        total_gt_pixels
    )

    # --------------------------------------------------------
    # Correct diagonal pixels
    # --------------------------------------------------------

    correct_by_class = []

    for row_index, gt_class in enumerate(
        gt_classes
    ):

        if gt_class in pred_classes:

            pred_index = pred_classes.index(
                gt_class
            )

            correct = matrix[
                row_index,
                pred_index
            ]

        else:

            correct = 0

        correct_by_class.append(
            correct
        )

    correct_by_class = np.asarray(
        correct_by_class,
        dtype=float
    )

    # --------------------------------------------------------
    # Total erroneous pixels
    # --------------------------------------------------------

    total_pixels = matrix.sum()

    total_correct = correct_by_class.sum()

    total_errors = (
        total_pixels -
        total_correct
    )

    # --------------------------------------------------------
    # Errors predicted AS each class
    # --------------------------------------------------------

    results = []

    for pred_index, pred_class in enumerate(
        pred_classes
    ):

        predicted_total = matrix[
            :,
            pred_index
        ].sum()

        # Correct pixels for this predicted class
        if pred_class in gt_classes:

            gt_index = gt_classes.index(
                pred_class
            )

            correct_pixels = matrix[
                gt_index,
                pred_index
            ]

        else:

            correct_pixels = 0

        erroneous_pixels_predicted_as_class = (
            predicted_total -
            correct_pixels
        )

        if total_errors > 0:

            error_share = (
                erroneous_pixels_predicted_as_class /
                total_errors
            )

        else:

            error_share = np.nan

        # Find GT abundance of corresponding class
        if pred_class in gt_classes:

            gt_index = gt_classes.index(
                pred_class
            )

            abundance = gt_abundance[
                gt_index
            ]

            gt_pixel_count = gt_totals[
                gt_index
            ]

        else:

            abundance = 0
            gt_pixel_count = 0

        # ----------------------------------------------------
        # Error attraction ratio
        # ----------------------------------------------------

        if abundance > 0:

            attraction_ratio = (
                error_share /
                abundance
            )

        else:

            attraction_ratio = np.nan

        results.append(
            {
                "biome":
                    biome_name,

                "class":
                    pred_class,

                "gt_pixels":
                    gt_pixel_count,

                "gt_abundance":
                    abundance,

                "gt_abundance_percent":
                    abundance * 100,

                "erroneous_pixels_predicted_as_class":
                    erroneous_pixels_predicted_as_class,

                "error_share":
                    error_share,

                "error_share_percent":
                    error_share * 100,

                "error_attraction_ratio":
                    attraction_ratio
            }
        )

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def calculate_correlation(
    results_df,
    biome_name,
    include_background=True
):

    temp = results_df.copy()

    if not include_background:

        temp = temp.loc[
            temp["class"].str.lower()
            !=
            BACKGROUND_CLASS.lower()
        ]

    temp = temp.dropna(
        subset=[
            "gt_abundance",
            "error_share"
        ]
    )

    # Remove zero-support classes
    temp = temp.loc[
        temp["gt_abundance"] > 0
    ]

    if len(temp) < 3:

        return {
            "biome":
                biome_name,

            "background_included":
                include_background,

            "n_classes":
                len(temp),

            "Spearman_rs":
                np.nan,

            "p_value":
                np.nan
        }

    rho, p = spearmanr(
        temp["gt_abundance"],
        temp["error_share"]
    )

    return {
        "biome":
            biome_name,

        "background_included":
            include_background,

        "n_classes":
            len(temp),

        "Spearman_rs":
            rho,

        "p_value":
            p
    }


# ============================================================
# PLOT
# ============================================================

def make_scatterplot(
    results_df,
    biome_name,
    include_background=True
):

    temp = results_df.copy()

    if not include_background:

        temp = temp.loc[
            temp["class"].str.lower()
            !=
            BACKGROUND_CLASS.lower()
        ]

    temp = temp.loc[
        temp["gt_abundance"] > 0
    ].copy()

    rho, p = spearmanr(
        temp["gt_abundance"],
        temp["error_share"]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        temp["gt_abundance_percent"],
        temp["error_share_percent"],
        s=40
    )

    # --------------------------------------------------------
    # Label classes
    # --------------------------------------------------------

    for _, row in temp.iterrows():

        ax.annotate(
            row["class"].replace(
                "_",
                " "
            ),
            (
                row["gt_abundance_percent"],
                row["error_share_percent"]
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8
        )

    # --------------------------------------------------------
    # 1:1 line
    # --------------------------------------------------------

    max_value = max(
        temp["gt_abundance_percent"].max(),
        temp["error_share_percent"].max()
    )

    ax.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
        linewidth=1
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    if p < 0.001:

        p_text = "p < 0.001"

    else:

        p_text = f"p = {p:.3f}"

    ax.text(
        0.04,
        0.96,
        f"$r_s$ = {rho:.2f}\n{p_text}",
        transform=ax.transAxes,
        va="top"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Ground-truth class abundance (%)"
    )

    ax.set_ylabel(
        "Share of classification errors (%)"
    )

    if include_background:

        title_suffix = "including Background"

    else:

        title_suffix = "excluding Background"

    ax.set_title(
        f"{biome_name} — {title_suffix}"
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    plt.tight_layout()

    filename_suffix = (
        "with_background"
        if include_background
        else
        "without_background"
    )

    output_path = (
        OUTPUT_DIR /
        f"{biome_name}_abundance_vs_errors_{filename_suffix}.png"
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    se_df = load_confusion_matrix(
        SE_NM_FILE
    )

    be_df = load_confusion_matrix(
        BE_LE_FILE
    )

    # --------------------------------------------------------
    # Analyse
    # --------------------------------------------------------

    se_results = analyse_confusion_matrix(
        se_df,
        "SE-NM"
    )

    be_results = analyse_confusion_matrix(
        be_df,
        "BE-LE"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined_results = pd.concat(
        [
            se_results,
            be_results
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save class-level results
    # --------------------------------------------------------

    combined_path = (
        OUTPUT_DIR /
        "class_abundance_error_attraction.csv"
    )

    combined_results.to_csv(
        combined_path,
        index=False
    )

    print(
        f"\nSaved class-level results:\n"
        f"{combined_path}"
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    correlation_results = []

    for biome_name, biome_df in [
        ("SE-NM", se_results),
        ("BE-LE", be_results)
    ]:

        correlation_results.append(
            calculate_correlation(
                biome_df,
                biome_name,
                include_background=True
            )
        )

        correlation_results.append(
            calculate_correlation(
                biome_df,
                biome_name,
                include_background=False
            )
        )

    correlation_df = pd.DataFrame(
        correlation_results
    )

    correlation_path = (
        OUTPUT_DIR /
        "abundance_error_spearman_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False
    )

    print(
        "\nSpearman correlations"
    )

    print(
        correlation_df.round(
            4
        ).to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n"
        f"{correlation_path}"
    )

    # --------------------------------------------------------
    # Attraction ratios
    # --------------------------------------------------------

    print(
        "\nClasses with strongest error attraction"
    )

    print(
        "=" * 70
    )

    for biome_name, biome_df in [
        ("SE-NM", se_results),
        ("BE-LE", be_results)
    ]:

        print(
            f"\n{biome_name}"
        )

        ranked = (
            biome_df
            .dropna(
                subset=[
                    "error_attraction_ratio"
                ]
            )
            .sort_values(
                "error_attraction_ratio",
                ascending=False
            )
        )

        print(
            ranked[
                [
                    "class",
                    "gt_abundance_percent",
                    "error_share_percent",
                    "error_attraction_ratio"
                ]
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    make_scatterplot(
        se_results,
        "SE-NM",
        include_background=True
    )

    make_scatterplot(
        se_results,
        "SE-NM",
        include_background=False
    )

    make_scatterplot(
        be_results,
        "BE-LE",
        include_background=True
    )

    make_scatterplot(
        be_results,
        "BE-LE",
        include_background=False
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
