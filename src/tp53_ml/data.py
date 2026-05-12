from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .preprocessing import clean_expression_columns


MUTATION_TYPE_MAP = {
    "Silent": "Synonymous",
    "Missense_Mutation": "Missense",
    "Nonsense_Mutation": "Nonsense",
    "Frame_Shift_Del": "Frameshift",
    "Frame_Shift_Ins": "Frameshift",
    "In_Frame_Del": "In-frame deletion",
    "In_Frame_Ins": "In-frame insertion",
    "Splice_Site": "Splice",
    "Translation_Start_Site": "Start lost",
    "Nonstop_Mutation": "Stop lost",
    "synonymous_variant": "Synonymous",
    "missense_variant": "Missense",
    "stop_gained": "Nonsense",
    "frameshift_variant": "Frameshift",
    "inframe_insertion": "In-frame insertion",
    "inframe_deletion": "In-frame deletion",
    "splice_donor_variant": "Splice",
    "splice_acceptor_variant": "Splice",
    "start_lost": "Start lost",
    "stop_lost": "Stop lost",
}

MUTATION_SEVERITY = {
    "Frameshift": 1,
    "Nonsense": 1,
    "Splice": 2,
    "Start lost": 2,
    "Stop lost": 2,
    "Missense": 3,
    "In-frame deletion": 4,
    "In-frame insertion": 4,
    "Synonymous": 5,
    "Other": 6,
}


DOWNLOAD_FILES_API = "https://depmap.org/portal/api/download/files"


def depmap_download_url(release: str, filename: str) -> str:
    files = pd.read_csv(DOWNLOAD_FILES_API)
    match = files[(files["release"] == release) & (files["filename"] == filename)]
    if match.empty:
        available = files.loc[files["release"] == release, "filename"].head(20).to_list()
        raise FileNotFoundError(
            f"{filename} was not found for {release}. First available files: {available}"
        )
    return str(match.iloc[0]["url"])


def download_depmap_file(release: str, filename: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 1024 and not _looks_like_html(out_path):
        return out_path
    urlretrieve(depmap_download_url(release, filename), out_path)
    if _looks_like_html(out_path):
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded HTML instead of data for {filename}.")
    return out_path


def try_download_candidates(release: str, filenames: list[str], out_dir: str | Path) -> Path:
    errors: list[str] = []
    for filename in filenames:
        try:
            path = download_depmap_file(release, filename, out_dir)
            if path.stat().st_size > 1024:
                return path
            errors.append(f"{filename}: downloaded file was unexpectedly small")
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    raise RuntimeError("Could not download any candidate file:\n" + "\n".join(errors))


def read_expression(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "IsDefaultEntryForModel" in df.columns:
        df = df.loc[df["IsDefaultEntryForModel"].astype(str).str.lower().eq("yes")]
    if "ModelID" in df.columns:
        df = df.set_index("ModelID")
    else:
        df = df.set_index(df.columns[0])
    metadata_cols = {
        "Unnamed: 0",
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors="ignore")
    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = clean_expression_columns(df)
    return df


def read_metadata(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def make_tp53_labels(mutations: pd.DataFrame, expression_index: pd.Index) -> pd.DataFrame:
    gene_col = _first_existing(mutations, ["HugoSymbol", "Hugo_Symbol", "gene_name", "Gene"])
    sample_col = _first_existing(
        mutations,
        ["ModelID", "Model_ID", "DepMap_ID", "Tumor_Sample_Barcode", "sample_id", "Sample"],
    )
    type_col = _first_existing(
        mutations,
        [
            "VariantClassification",
            "Variant_Classification",
            "MolecularConsequence",
            "VariantInfo",
            "var_class",
            "Variant_annotation",
        ],
        required=False,
    )
    if gene_col is None or sample_col is None:
        raise ValueError("Mutation table must contain recognizable gene and sample identifier columns.")

    tp53 = mutations.loc[mutations[gene_col].astype(str).str.upper() == "TP53"].copy()
    if "IsDefaultEntryForModel" in tp53.columns:
        tp53 = tp53.loc[tp53["IsDefaultEntryForModel"].astype(str).str.lower().eq("yes")]
    tp53[sample_col] = tp53[sample_col].astype(str)
    if type_col is None:
        tp53["mutation_type"] = "Other"
    else:
        tp53["mutation_type"] = tp53[type_col].map(classify_variant)

    best_type = {}
    for sample, group in tp53.groupby(sample_col):
        ranked = sorted(
            group["mutation_type"].astype(str).to_list(),
            key=lambda value: MUTATION_SEVERITY.get(value, MUTATION_SEVERITY["Other"]),
        )
        best_type[sample] = ranked[0] if ranked else "Other"

    labels = pd.DataFrame({"sample_id": expression_index.astype(str)})
    labels["tp53_mutant"] = labels["sample_id"].isin(best_type).astype(int)
    labels["mutation_type"] = labels["sample_id"].map(best_type).fillna("WT")
    labels["mutation_type_collapsed"] = labels["mutation_type"].replace(
        {
            "In-frame deletion": "In-frame indel",
            "In-frame insertion": "In-frame indel",
            "Splice": "Other",
            "Start lost": "Other",
            "Stop lost": "Other",
        }
    )
    return labels


def align_metadata(metadata: pd.DataFrame, sample_ids: pd.Index) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame({"sample_id": sample_ids.astype(str)})
    id_col = _first_existing(metadata, ["ModelID", "Model_ID", "DepMap_ID", "sample_id", "Sample"], required=False)
    if id_col is None:
        return pd.DataFrame({"sample_id": sample_ids.astype(str)})
    metadata = metadata.copy()
    metadata[id_col] = metadata[id_col].astype(str)
    metadata = metadata.drop_duplicates(id_col).set_index(id_col)
    out = metadata.reindex(sample_ids.astype(str)).reset_index(names="sample_id")
    return out


def _first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def classify_variant(value) -> str:
    if pd.isna(value):
        return "Other"
    text = str(value)
    if text in MUTATION_TYPE_MAP:
        return MUTATION_TYPE_MAP[text]
    terms = [part.strip() for part in text.replace("&", ",").split(",")]
    for term in terms:
        if term in MUTATION_TYPE_MAP:
            return MUTATION_TYPE_MAP[term]
    lowered = text.lower()
    if "frameshift" in lowered:
        return "Frameshift"
    if "stop_gained" in lowered or "nonsense" in lowered:
        return "Nonsense"
    if "splice" in lowered:
        return "Splice"
    if "missense" in lowered:
        return "Missense"
    if "inframe_deletion" in lowered:
        return "In-frame deletion"
    if "inframe_insertion" in lowered:
        return "In-frame insertion"
    if "synonymous" in lowered or "silent" in lowered:
        return "Synonymous"
    return "Other"


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(256).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")
