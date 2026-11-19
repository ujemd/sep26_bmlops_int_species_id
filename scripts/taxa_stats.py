"""Research-grade and openly licensed observation counts per taxon on iNaturalist.
Evidence for the perimeter in params.yaml (well-represented taxa). Writes reports/taxa_stats.md.

Usage (in the Airflow container, or `make taxa-stats`): python -m scripts.taxa_stats
"""
from src import config
from src.data.ingest import api_get, resolve_taxon


def main():
    p = config.params()["data"]
    rows = ["| species | research-grade observations | openly licensed photos |", "|---|---|---|"]
    for name in p["taxa"]:
        taxon_id, canonical = resolve_taxon(name)
        query = dict(taxon_id=taxon_id, quality_grade="research", photos="true", per_page=0)
        total = api_get("observations", **query)["total_results"]
        open_ = api_get("observations", photo_license=",".join(p["allowed_licenses"]), **query)["total_results"]
        rows.append(f"| *{canonical}* | {total:,} | {open_:,} |")
        print(rows[-1], flush=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "taxa_stats.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
