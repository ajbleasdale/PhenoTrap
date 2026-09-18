from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

IN_XLSX = r"path-to-annotated_dataset.xlsx"
OUT_DIR = r"path-to-partition_output"

COL_DEPLOY = "deployment_id"
COL_DATE = "date"

# Final desired proportions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

RANDOM_SEED = 42

# How strongly to prioritise similarity in monthly distribution
# relative to achieving the target number of images.
MONTH_WEIGHT = 0.35

# Acceptable deviation from target image count
TOLERANCE_ROWS = 10


# =========================================================
# HELPERS
# =========================================================

def get_month_series(df):
    """Convert the date column to YYYY-MM month values."""
    if COL_DATE not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    dates = pd.to_datetime(df[COL_DATE], errors="coerce")

    return dates.apply(
        lambda x: f"{x.year:04d}-{x.month:02d}"
        if pd.notna(x) else None
    )


def split_by_deployment(
    df,
    target_frac,
    seed=42,
    month_weight=0.35,
    tolerance=10,
):
    """
    Divide a dataframe into two groups while ensuring that
    each deployment occurs in only one partition.

    The first returned dataframe aims to contain target_frac
    of the images.

    Deployment allocation considers:
      1. target image count
      2. similarity in monthly distribution
    """

    df = df.copy()

    if COL_DEPLOY not in df.columns:
        raise ValueError(
            f"Required column '{COL_DEPLOY}' not found."
        )

    df["_month"] = get_month_series(df)

    n_total = len(df)
    target_n = round(n_total * target_frac)

    # -----------------------------------------------------
    # Overall monthly distribution
    # -----------------------------------------------------

    if df["_month"].notna().any():

        month_counts = df["_month"].value_counts()

        month_props = (
            month_counts / month_counts.sum()
        ).to_dict()

        use_months = True

    else:

        month_props = {}
        use_months = False


    # -----------------------------------------------------
    # Deployment statistics
    # -----------------------------------------------------

    grouped = df.groupby(COL_DEPLOY)

    dep_sizes = grouped.size().to_dict()

    if use_months:

        dep_month_counts = (
            grouped["_month"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

    else:

        dep_month_counts = {
            dep: {} for dep in dep_sizes
        }


    # -----------------------------------------------------
    # Deployment order
    # -----------------------------------------------------

    rng = np.random.default_rng(seed)

    deployments = list(dep_sizes.keys())

    rng.shuffle(deployments)

    # Larger deployments first
    deployments.sort(
        key=lambda d: dep_sizes[d],
        reverse=True
    )


    selected = set()
    remaining = set()

    selected_count = 0

    selected_month_counts = {
        month: 0 for month in month_props
    }


    # -----------------------------------------------------
    # Allocation score
    # -----------------------------------------------------

    def score_if_added(dep):

        new_count = selected_count + dep_sizes[dep]

        size_penalty = abs(
            target_n - new_count
        )

        if not use_months:
            return size_penalty

        temp_months = selected_month_counts.copy()

        for month, count in dep_month_counts.get(
            dep, {}
        ).items():

            temp_months[month] = (
                temp_months.get(month, 0)
                + count
            )

        total_month_images = sum(
            temp_months.values()
        )

        if total_month_images == 0:
            month_penalty = 0

        else:

            month_penalty = sum(

                abs(
                    temp_months.get(month, 0)
                    / total_month_images
                    - expected_prop
                )

                for month, expected_prop
                in month_props.items()
            )

        return (
            (1 - month_weight) * size_penalty
            + month_weight
            * month_penalty
            * n_total
        )


    # -----------------------------------------------------
    # Greedy deployment assignment
    # -----------------------------------------------------

    for dep in deployments:

        if selected_count >= target_n:

            remaining.add(dep)
            continue

        add_score = score_if_added(dep)

        skip_score = abs(
            target_n - selected_count
        )

        if add_score <= skip_score:

            selected.add(dep)

            selected_count += dep_sizes[dep]

            for month, count in dep_month_counts.get(
                dep, {}
            ).items():

                selected_month_counts[month] = (
                    selected_month_counts.get(
                        month, 0
                    )
                    + count
                )

        else:

            remaining.add(dep)


    # -----------------------------------------------------
    # Repair if far from target
    # -----------------------------------------------------

    def selected_size():

        return sum(
            dep_sizes[d]
            for d in selected
        )


    current_n = selected_size()

    if abs(current_n - target_n) > tolerance:

        if current_n < target_n:

            candidates = sorted(
                remaining,
                key=lambda d: dep_sizes[d]
            )

            for dep in candidates:

                selected.add(dep)
                remaining.remove(dep)

                current_n = selected_size()

                if current_n >= target_n:
                    break

        else:

            candidates = sorted(
                selected,
                key=lambda d: dep_sizes[d]
            )

            for dep in candidates:

                selected.remove(dep)
                remaining.add(dep)

                current_n = selected_size()

                if current_n <= target_n:
                    break


    selected_df = df[
        df[COL_DEPLOY].isin(selected)
    ].copy()

    remaining_df = df[
        df[COL_DEPLOY].isin(remaining)
    ].copy()


    selected_df.drop(
        columns=["_month"],
        inplace=True
    )

    remaining_df.drop(
        columns=["_month"],
        inplace=True
    )


    # -----------------------------------------------------
    # Leakage check
    # -----------------------------------------------------

    overlap = (

        set(
            selected_df[COL_DEPLOY]
        )

        &

        set(
            remaining_df[COL_DEPLOY]
        )
    )

    if overlap:

        raise RuntimeError(
            "Deployment leakage detected."
        )


    return selected_df, remaining_df


# =========================================================
# MAIN PARTITION PROCEDURE
# =========================================================

def main():

    if not np.isclose(
        TRAIN_FRAC + VAL_FRAC + TEST_FRAC,
        1
    ):
        raise ValueError(
            "TRAIN_FRAC + VAL_FRAC + TEST_FRAC "
            "must equal 1."
        )


    input_path = Path(IN_XLSX)

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )


    df = pd.read_excel(input_path)


    # =====================================================
    # STEP 1
    # Create TEST set
    # =====================================================

    test_df, development_df = (
        split_by_deployment(
            df,
            target_frac=TEST_FRAC,
            seed=RANDOM_SEED,
            month_weight=MONTH_WEIGHT,
            tolerance=TOLERANCE_ROWS,
        )
    )


    # =====================================================
    # STEP 2
    # Create VALIDATION from remaining development data
    # =====================================================

    # Need to convert desired overall validation fraction
    # into a fraction of the remaining data.

    remaining_fraction = (
        TRAIN_FRAC + VAL_FRAC
    )

    validation_within_remaining = (
        VAL_FRAC / remaining_fraction
    )


    val_df, train_df = (
        split_by_deployment(
            development_df,
            target_frac=validation_within_remaining,
            seed=RANDOM_SEED + 1,
            month_weight=MONTH_WEIGHT,
            tolerance=TOLERANCE_ROWS,
        )
    )


    # =====================================================
    # FINAL LEAKAGE CHECK
    # =====================================================

    train_dep = set(
        train_df[COL_DEPLOY]
    )

    val_dep = set(
        val_df[COL_DEPLOY]
    )

    test_dep = set(
        test_df[COL_DEPLOY]
    )


    if train_dep & val_dep:
        raise RuntimeError(
            "Train-validation deployment leakage."
        )

    if train_dep & test_dep:
        raise RuntimeError(
            "Train-test deployment leakage."
        )

    if val_dep & test_dep:
        raise RuntimeError(
            "Validation-test deployment leakage."
        )


    # =====================================================
    # OUTPUT
    # =====================================================

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    train_df.to_excel(
        out_dir / "train.xlsx",
        index=False
    )

    val_df.to_excel(
        out_dir / "validation.xlsx",
        index=False
    )

    test_df.to_excel(
        out_dir / "test.xlsx",
        index=False
    )


    # =====================================================
    # REPORT
    # =====================================================

    total = len(df)

    print("\n=== FINAL PARTITION ===")

    print(
        f"Total images: {total}"
    )

    print(
        f"Training:   {len(train_df)} "
        f"({len(train_df)/total:.1%})"
    )

    print(
        f"Validation: {len(val_df)} "
        f"({len(val_df)/total:.1%})"
    )

    print(
        f"Testing:    {len(test_df)} "
        f"({len(test_df)/total:.1%})"
    )


    print("\n=== DEPLOYMENTS ===")

    print(
        f"Training deployments: "
        f"{train_df[COL_DEPLOY].nunique()}"
    )

    print(
        f"Validation deployments: "
        f"{val_df[COL_DEPLOY].nunique()}"
    )

    print(
        f"Testing deployments: "
        f"{test_df[COL_DEPLOY].nunique()}"
    )


    print(
        "\nNo deployment leakage detected."
    )


if __name__ == "__main__":
    main()
