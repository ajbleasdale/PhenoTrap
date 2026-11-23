
"""
VISUALISE PATCH SIZE VS SEGMENTATION PERFORMANCE
=================================================

Input:
    patch_level_performance.csv

Outputs:
    - overall patch size figure
    - class-specific patch-size figure
    - size-category figure
    - class-specific Spearman statistics CSV
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import (
    spearmanr
)


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = Path(
    r"\\storage-ume.slu.se\home$\arbl0002\My Documents\1_Active\Segmentation\patch_level_performance.csv"
)

OUTPUT_DIR = Path(
    r"path-to-output-directory"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Optional minimum patch size
# ------------------------------------------------------------

# Useful for removing single-pixel annotation artefacts.
#
# Initially inspect both:
# 1 pixel
# 10 pixels
# 25 pixels
# 50 pixels

MIN_PATCH_SIZE = 1


# ------------------------------------------------------------
# Minimum patches required to calculate class correlation
# ------------------------------------------------------------

MIN_PATCHES_PER_CLASS = 20


# ------------------------------------------------------------
# Font
# ------------------------------------------------------------

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 10

plt.rcParams[
    "axes.labelsize"
] = 11

plt.rcParams[
    "xtick.labelsize"
] = 9

plt.rcParams[
    "ytick.labelsize"
] = 9


POINT_SIZE = 12
POINT_ALPHA = 0.25


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

df = df.loc[
    df[
        "patch_area_pixels"
    ]
    >=
    MIN_PATCH_SIZE
].copy()

print(
    f"Patches retained: "
    f"{len(df):,}"
)


# ============================================================
# OVERALL CORRELATION
# ============================================================

rho, p = spearmanr(
    df[
        "patch_area_pixels"
    ],
    df[
        "patch_accuracy"
    ]
)

print(
    "\nOverall relationship"
)

print(
    f"Spearman rs = "
    f"{rho:.3f}"
)

print(
    f"p = {p:.4g}"
)


# ============================================================
# FIGURE 1 — OVERALL PATCH SIZE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 4.5)
)

ax.scatter(
    df[
        "patch_area_pixels"
    ],
    df[
        "patch_accuracy"
    ],
    s=POINT_SIZE,
    alpha=POINT_ALPHA,
    edgecolors="none"
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Ground-truth patch area (pixels)"
)

ax.set_ylabel(
    "Patch accuracy"
)

ax.set_ylim(
    0,
    1.02
)

if p < 0.001:

    p_text = (
        "p < 0.001"
    )

else:

    p_text = (
        f"p = {p:.3f}"
    )

ax.text(
    0.04,
    0.96,
    (
        f"$r_s$ = {rho:.2f}\n"
        f"{p_text}"
    ),
    transform=ax.transAxes,
    va="top"
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "overall_patch_size_vs_accuracy.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR /
    "overall_patch_size_vs_accuracy.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CLASS-SPECIFIC SPEARMAN CORRELATIONS
# ============================================================

class_results = []

for class_name, group in df.groupby(
    "gt_class_name"
):

    group = group.dropna(
        subset=[
            "patch_area_pixels",
            "patch_accuracy"
        ]
    )

    if (
        len(group)
        <
        MIN_PATCHES_PER_CLASS
    ):

        continue

    class_rho, class_p = (
        spearmanr(
            group[
                "patch_area_pixels"
            ],
            group[
                "patch_accuracy"
            ]
        )
    )

    class_results.append(
        {
            "class":
                class_name,

            "n_patches":
                len(group),

            "median_patch_area":
                group[
                    "patch_area_pixels"
                ].median(),

            "median_patch_accuracy":
                group[
                    "patch_accuracy"
                ].median(),

            "Spearman_rs":
                class_rho,

            "p_value":
                class_p
        }
    )


class_stats = pd.DataFrame(
    class_results
)

class_stats = class_stats.sort_values(
    "Spearman_rs",
    ascending=False
)

class_stats.to_csv(
    OUTPUT_DIR /
    "class_patch_size_correlations.csv",
    index=False
)


print(
    "\nClass-specific relationships"
)

print(
    class_stats.round(
        3
    ).to_string(
        index=False
    )
)


# ============================================================
# FIGURE 2 — CLASS-SPECIFIC RELATIONSHIPS
# ============================================================

classes = (
    class_stats[
        "class"
    ].tolist()
)

n_classes = len(
    classes
)

n_cols = 3

n_rows = int(
    np.ceil(
        n_classes /
        n_cols
    )
)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        10,
        3.2 * n_rows
    ),
    sharey=True
)

axes = np.array(
    axes
).reshape(-1)


for i, class_name in enumerate(
    classes
):

    ax = axes[i]

    group = df.loc[
        df[
            "gt_class_name"
        ]
        ==
        class_name
    ]

    ax.scatter(
        group[
            "patch_area_pixels"
        ],
        group[
            "patch_accuracy"
        ],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none"
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0,
        1.02
    )

    stats_row = (
        class_stats.loc[
            class_stats[
                "class"
            ]
            ==
            class_name
        ]
        .iloc[0]
    )

    class_rho = (
        stats_row[
            "Spearman_rs"
        ]
    )

    class_p = (
        stats_row[
            "p_value"
        ]
    )

    if class_p < 0.001:

        p_text = (
            "p < 0.001"
        )

    else:

        p_text = (
            f"p = {class_p:.3f}"
        )

    ax.text(
        0.04,
        0.05,
        (
            f"$r_s$ = "
            f"{class_rho:.2f}\n"
            f"{p_text}"
        ),
        transform=ax.transAxes,
        va="bottom"
    )

    ax.set_title(
        class_name.replace(
            "_",
            " "
        )
    )

    ax.set_xlabel(
        "Patch area (pixels)"
    )

    if (
        i %
        n_cols
        ==
        0
    ):

        ax.set_ylabel(
            "Patch accuracy"
        )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# Remove unused axes
for j in range(
    n_classes,
    len(axes)
):

    axes[j].axis(
        "off"
    )


plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "patch_size_vs_accuracy_by_class.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR /
    "patch_size_vs_accuracy_by_class.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SMALL / MEDIUM / LARGE PATCHES
# ============================================================

def assign_size_category(
    group
):
    """
    Define patch size relative to other patches of the SAME class.

    Lower third  = Small
    Middle third = Medium
    Upper third  = Large
    """

    group = group.copy()

    # rank first so identical values do not cause qcut failure
    ranks = group[
        "patch_area_pixels"
    ].rank(
        method="first"
    )

    group[
        "size_category"
    ] = pd.qcut(
        ranks,
        q=3,
        labels=[
            "Small",
            "Medium",
            "Large"
        ]
    )

    return group


df_binned = (
    df.groupby(
        "gt_class_name",
        group_keys=False
    )
    .apply(
        assign_size_category
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SUMMARISE SIZE GROUPS
# ============================================================

size_summary = (
    df_binned
    .groupby(
        [
            "gt_class_name",
            "size_category"
        ],
        observed=True
    )
    .agg(
        n_patches=(
            "patch_accuracy",
            "size"
        ),

        median_patch_area=(
            "patch_area_pixels",
            "median"
        ),

        mean_accuracy=(
            "patch_accuracy",
            "mean"
        ),

        median_accuracy=(
            "patch_accuracy",
            "median"
        )
    )
    .reset_index()
)


size_summary.to_csv(
    OUTPUT_DIR /
    "patch_size_category_summary.csv",
    index=False
)


# ============================================================
# FIGURE 3 — SMALL VS MEDIUM VS LARGE
# ============================================================

plot_classes = (
    df_binned[
        "gt_class_name"
    ]
    .value_counts()
)

plot_classes = (
    plot_classes[
        plot_classes
        >=
        MIN_PATCHES_PER_CLASS
    ]
    .index
    .tolist()
)


n_classes = len(
    plot_classes
)

n_cols = 3

n_rows = int(
    np.ceil(
        n_classes /
        n_cols
    )
)


fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        10,
        3.2 * n_rows
    ),
    sharey=True
)

axes = np.array(
    axes
).reshape(-1)


category_order = [
    "Small",
    "Medium",
    "Large"
]


for i, class_name in enumerate(
    plot_classes
):

    ax = axes[i]

    group = df_binned.loc[
        df_binned[
            "gt_class_name"
        ]
        ==
        class_name
    ]

    box_data = []

    for category in category_order:

        values = group.loc[
            group[
                "size_category"
            ]
            ==
            category,
            "patch_accuracy"
        ].dropna()

        box_data.append(
            values.to_numpy()
        )

    ax.boxplot(
        box_data,
        labels=category_order,
        showfliers=False
    )

    # Individual points
    rng = np.random.default_rng(
        42
    )

    for position, category in enumerate(
        category_order,
        start=1
    ):

        values = group.loc[
            group[
                "size_category"
            ]
            ==
            category,
            "patch_accuracy"
        ].dropna().to_numpy()

        jitter = rng.normal(
            0,
            0.05,
            len(values)
        )

        ax.scatter(
            position + jitter,
            values,
            s=8,
            alpha=0.2,
            edgecolors="none"
        )

    ax.set_ylim(
        0,
        1.02
    )

    ax.set_title(
        class_name.replace(
            "_",
            " "
        )
    )

    ax.set_xlabel(
        "Relative patch size"
    )

    if (
        i %
        n_cols
        ==
        0
    ):

        ax.set_ylabel(
            "Patch accuracy"
        )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


for j in range(
    n_classes,
    len(axes)
):

    axes[j].axis(
        "off"
    )


plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "accuracy_by_relative_patch_size.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR /
    "accuracy_by_relative_patch_size.pdf",
    bbox_inches="tight"
)

plt.show()
