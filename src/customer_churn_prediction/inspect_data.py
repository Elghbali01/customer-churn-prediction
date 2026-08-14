"""Generate a read-only initial inspection of the IBM Telco churn CSV."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/Telco-Customer-Churn.csv")
DEFAULT_REPORT = Path("reports/data-understanding.md")
TARGET = "Churn"
IDENTIFIER = "customerID"


def _markdown_table(frame: pd.DataFrame) -> str:
    rendered = frame.reset_index()
    headers = [str(column) for column in rendered.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in rendered.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(data: pd.DataFrame, source: Path) -> str:
    """Build a Markdown report without transforming the supplied dataframe."""
    info_buffer = io.StringIO()
    data.info(buf=info_buffer, show_counts=True)
    object_columns = data.select_dtypes(include="object").columns
    blank_counts = data[object_columns].apply(lambda col: col.str.strip().eq("").sum())
    target_counts = data[TARGET].value_counts(dropna=False).rename("count").to_frame()
    target_counts["proportion"] = target_counts["count"] / len(data)

    sections = [
        "# Étape 2 — Rapport de Data Understanding",
        "",
        "> Ce rapport détecte les caractéristiques et problèmes du fichier brut. "
        "Il n'applique aucun nettoyage ni preprocessing.",
        "",
        "## Périmètre et source inspectée",
        "",
        f"- Fichier : `{source.as_posix()}`",
        f"- Dimensions : **{data.shape[0]} lignes × {data.shape[1]} colonnes**",
        f"- Identifiant candidat : `{IDENTIFIER}`",
        f"- Target : `{TARGET}`",
        "",
        "## Colonnes",
        "",
        ", ".join(f"`{column}`" for column in data.columns),
        "",
        "## Premières observations",
        "",
        _markdown_table(data.head()),
        "",
        "## Types Pandas observés",
        "",
        _markdown_table(data.dtypes.astype(str).rename("dtype").to_frame()),
        "",
        "## Informations générales (`DataFrame.info`)",
        "",
        "```text",
        info_buffer.getvalue().rstrip(),
        "```",
        "",
        "## Statistiques descriptives",
        "",
        "### Variables numériques",
        "",
        _markdown_table(data.describe().transpose()),
        "",
        "### Variables non numériques",
        "",
        _markdown_table(data.describe(include="object").transpose()),
        "",
        "## Nombre de valeurs uniques",
        "",
        _markdown_table(data.nunique(dropna=False).rename("n_unique_including_na").to_frame()),
        "",
        "## Valeurs manquantes et chaînes vides",
        "",
        _markdown_table(
            pd.DataFrame(
                {
                    "missing_detected_by_pandas": data.isna().sum(),
                    "blank_or_whitespace_strings": blank_counts.reindex(data.columns, fill_value=0),
                }
            )
        ),
        "",
        "## Distribution brute de la target",
        "",
        _markdown_table(target_counts),
        "",
        "## Doublons potentiels",
        "",
        f"- Lignes entièrement dupliquées : **{int(data.duplicated().sum())}**",
        f"- Valeurs dupliquées de `{IDENTIFIER}` : **{int(data[IDENTIFIER].duplicated().sum())}**",
        "",
        "## Observations",
        "",
        "- `TotalCharges` est lu comme `object`, bien qu'il représente des montants.",
        f"- `TotalCharges` contient **{int(blank_counts['TotalCharges'])}** chaînes vides ou composées d'espaces ; "
        "Pandas ne les compte pas comme valeurs manquantes avec la lecture utilisée.",
        "- `SeniorCitizen` est lu comme entier avec deux valeurs distinctes ; son rôle semble être celui d'un indicateur catégoriel.",
        "- Certaines colonnes de services possèdent des catégories telles que `No internet service` ou `No phone service`, distinctes de `No`.",
        "- La target contient deux modalités brutes, `Yes` et `No`.",
        "- Le fichier sélectionné ne contient pas les champs enrichis connus `Churn Score`, `Churn Reason`, "
        "`Churn Category`, `Customer Status` ou `Churn Value`.",
        "",
        "## Décisions futures — non appliquées à cette étape",
        "",
        "- Définir le traitement de `TotalCharges` et de ses chaînes vides lors du Data Cleaning.",
        "- Confirmer le typage métier de `SeniorCitizen`.",
        "- Décider du traitement de l'identifiant avant la modélisation.",
        "- Définir et documenter l'encodage de la target et des catégories lors du preprocessing.",
        "- Réévaluer toute colonne candidate au regard de sa disponibilité au moment de la prédiction et du target leakage.",
        "",
    ]
    return "\n".join(sections)


def inspect(source: Path = DEFAULT_INPUT, report: Path = DEFAULT_REPORT) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source}")
    data = pd.read_csv(source)
    if TARGET not in data.columns or IDENTIFIER not in data.columns:
        raise ValueError(f"Expected columns missing: {TARGET!r} and/or {IDENTIFIER!r}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(data, source), encoding="utf-8")
    print(f"Inspection report written to {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    inspect(args.input, args.report)


if __name__ == "__main__":
    main()
