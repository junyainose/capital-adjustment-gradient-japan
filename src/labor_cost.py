"""Imputing firm labour costs and value added from national accounting data.

Why this is needed
------------------
Japanese income statements have no labour cost line.  In manufacturing, wages of
production workers enter cost of goods sold; only the non-production share
appears in selling and administrative expenses.  The cost of sales schedule that
would separate the two is not required on a consolidated basis, and a growing
number of firms omit it from the parent-only statements as well.

Labour cost is therefore imputed from the Financial Statements Statistics of
Corporations by Industry, published by the Ministry of Finance, which reports
compensation and headcount by industry and capital size:

    labour cost_j = w(industry, year) x employees_j
    value added_j ~ operating profit_j + depreciation_j + labour cost_j

Why this source rather than the Basic Survey on Wage Structure: the concept
matches.  The Financial Statements Statistics aggregate firms' own accounts, so
"compensation as recorded in the income statement" means the same thing there as
in NEEDS.  The Basic Survey is an establishment survey of payments to workers and
excludes statutory benefits and retirement provisions.  The Financial Statements
Statistics also break out capital size, so listed firms can be matched to the
stratum of at least 100 million yen of paid-in capital.

Scope
-----
The paper's main results do not depend on this module.  MRPK is measured with
operating surplus, operating profit plus depreciation, which is what the model's
profit function implies and which requires no outside data.  Value added is
reported only as a robustness check, because it reproduces the exact numerator
used in the misallocation literature and by Albrizio, Gonzalez and Khametshin
(2026).  Anyone reproducing the main tables can ignore this file.

Caveat
------
Compensation in the Financial Statements Statistics is directors' pay and
bonuses, employees' pay and bonuses, and welfare expenses.  Where welfare
expenses are not among the items downloaded, the imputation is built from pay
and bonuses alone and understates labour cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Names of the 32 Nikkei medium industry classifications.  Reconstructed from
# representative firms, since the classification file carries codes only.
# Financial industries (banks, securities, insurance) are absent by construction.
NKILM_NAME: dict[str, str] = {
    "01": "食品",             "03": "繊維",           "05": "パルプ・紙",
    "07": "化学",             "09": "医薬品",         "11": "石油",
    "13": "ゴム",             "15": "窯業",           "17": "鉄鋼",
    "19": "非鉄金属・金属製品",  "21": "機械",           "23": "電気機器",
    "25": "造船",             "27": "自動車・部品",    "29": "その他輸送用機器",
    "31": "精密機器",          "33": "その他製造",      "35": "水産・農林",
    "37": "鉱業",             "41": "建設",           "43": "商社・卸売",
    "45": "小売",             "53": "不動産",         "55": "鉄道・バス",
    "57": "陸運",             "59": "海運",           "61": "空運",
    "63": "倉庫・運輸関連",     "65": "情報・通信",      "67": "電力",
    "69": "ガス",             "71": "サービス",
}

NKILM_NAME_EN: dict[str, str] = {
    "01": "Food", "03": "Textiles", "05": "Pulp and paper",
    "07": "Chemicals", "09": "Pharmaceuticals", "11": "Petroleum",
    "13": "Rubber", "15": "Ceramics and glass", "17": "Steel",
    "19": "Nonferrous and fabricated metals", "21": "Machinery",
    "23": "Electrical machinery", "25": "Shipbuilding",
    "27": "Motor vehicles and parts", "29": "Other transport equipment",
    "31": "Precision instruments", "33": "Other manufacturing",
    "35": "Fisheries and forestry", "37": "Mining", "41": "Construction",
    "43": "Trading and wholesale", "45": "Retail", "53": "Real estate",
    "55": "Railways and buses", "57": "Road transport", "59": "Shipping",
    "61": "Air transport", "63": "Warehousing and logistics",
    "65": "Information and communications", "67": "Electric power",
    "69": "Gas", "71": "Services",
}

# Mapping from the Nikkei medium classification to industries in the Financial
# Statements Statistics.  The values are the Japanese industry labels as they
# appear in the downloaded file and must not be translated.  Where several are
# listed, they are averaged with headcount weights.
#
# This mapping was rebuilt from scratch in August 2026.  The earlier version was
# offset by one position throughout -- 19 was assigned to steel, 21 to
# nonferrous metals, 23 to fabricated metals, and so on -- so every
# manufacturing wage was taken from the neighbouring industry.  The current
# version was verified by identifying representative firms in each code.
#
# Only industries covered for the whole sample are used.  Revisions to the Japan
# Standard Industrial Classification mean that general-purpose machinery,
# information and communication equipment (from 2004), pure holding companies,
# and scientific and professional services exist only from 2009, and splicing a
# series in mid-sample introduces a step in the imputed wage.  Textiles are the
# one exception and are handled by SPLICE below.
NKILM_TO_HOJIN: dict[str, list[str]] = {
    # --- manufacturing ---
    "01": ["食料品製造業"],
    "03": ["繊維工業"],                       # spliced to the old series pre-2009
    "05": ["パルプ・紙・紙加工品製造業"],
    "07": ["化学工業"],
    "09": ["化学工業"],                       # pharmaceuticals not reported separately
    "11": ["石油製品・石炭製品製造業"],
    "13": ["その他の製造業"],                 # rubber not reported separately
    "15": ["窯業・土石製品製造業"],
    "17": ["鉄鋼業"],
    "19": ["非鉄金属製造業", "金属製品製造業"],
    "21": ["生産用機械器具製造業"],           # general-purpose machinery is 2009 onwards
    "23": ["電気機械器具製造業"],             # ICT equipment is 2004 onwards
    "25": ["その他の輸送用機械器具製造業"],   # shipbuilding
    "27": ["自動車・同附属品製造業"],
    "29": ["その他の輸送用機械器具製造業"],
    "31": ["業務用機械器具製造業"],           # precision instruments
    "33": ["その他の製造業", "印刷・同関連業"],
    # --- non-manufacturing ---
    "35": ["農林水産業(集約)"],
    "37": ["鉱業、採石業、砂利採取業"],
    "41": ["建設業"],
    "43": ["卸売業"],
    "45": ["小売業"],
    "53": ["不動産業"],
    "55": ["陸運業"],                         # railways and buses
    "57": ["陸運業"],
    "59": ["水運業"],
    "61": ["その他の運輸業"],                 # air transport
    "63": ["その他の運輸業"],                 # warehousing
    "65": ["情報通信業"],
    "67": ["電気業"],
    "69": ["ガス・熱供給・水道業"],
    "71": ["サービス業(集約)"],
}

# Series broken by classification revisions.  new name: (old name, last year of
# the old series).
SPLICE: dict[str, tuple[str, int]] = {
    "繊維工業": ("繊維工業(H20年度まで)", 2008),
}

FALLBACK_INDUSTRY = "全産業（除く金融保険業）"   # all industries ex. finance
SIZE_LABEL = "1億円以上"                        # paid-in capital >= 100m yen


def load_hojin(path: str | Path, encoding: str = "cp932",
               skiprows: int = 10) -> pd.DataFrame:
    """Read the Financial Statements Statistics extract downloaded from e-Stat.

    The file is Shift-JIS encoded with a ten-row preamble and no usable header
    row, so columns are assigned by position.  The expected item set is sales,
    directors' pay and bonuses, employees' pay and bonuses, average numbers of
    directors and employees, and value added.

    Suppressed cells arrive as "***", "*" or "-" and are read as missing.
    """
    columns = ["ind_code", "ind_sub", "industry", "size_code", "size_sub", "size",
               "yr_code", "yr_sub", "year", "item", "sales", "exec_pay",
               "exec_bonus", "emp_pay", "emp_bonus", "n_exec", "n_emp",
               "value_added"]
    df = pd.read_csv(path, encoding=encoding, skiprows=skiprows, header=None)
    df.columns = columns[:df.shape[1]]

    def to_num(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace(",", "")
        return pd.to_numeric(s.replace({"***": np.nan, "*": np.nan, "-": np.nan}),
                             errors="coerce")

    for c in ["sales", "exec_pay", "exec_bonus", "emp_pay", "emp_bonus",
              "n_exec", "n_emp", "value_added"]:
        if c in df.columns:
            df[c] = to_num(df[c])
    df["fy"] = pd.to_numeric(
        df["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    return df.dropna(subset=["fy"])


def wage_by_industry(hojin: pd.DataFrame, size: str = SIZE_LABEL,
                     include_exec: bool = False) -> pd.DataFrame:
    """Compensation per head by industry and year, in millions of yen.

    `include_exec` adds directors' pay to the numerator and directors to the
    denominator.  Directors are a small share of headcount at listed firms, so
    the default excludes them; the choice moves the imputed wage by little.
    """
    h = hojin[hojin["size"] == size].copy()
    if include_exec:
        pay = (h["exec_pay"].fillna(0) + h["exec_bonus"].fillna(0)
               + h["emp_pay"].fillna(0) + h["emp_bonus"].fillna(0))
        heads = h["n_emp"].fillna(0) + h["n_exec"].fillna(0)
    else:
        pay = h["emp_pay"].fillna(0) + h["emp_bonus"].fillna(0)
        heads = h["n_emp"]
    h["wage_per_head"] = np.where(heads > 0, pay / heads, np.nan)
    return h[["industry", "fy", "wage_per_head", "n_emp", "sales",
              "value_added"]].dropna(subset=["wage_per_head"])


def build_wage_map(wage: pd.DataFrame,
                   mapping: dict[str, list[str]] | None = None,
                   splice: dict[str, tuple[str, int]] | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Expand the industry wage series onto the Nikkei classification.

    Where a Nikkei code maps to several source industries, the wage is averaged
    with headcount weights.  Series broken by classification revisions are
    joined to their predecessors as specified in `splice`.

    Returns
    -------
    DataFrame with columns [nkilm, fy, wage_per_head].
    """
    mapping = mapping or NKILM_TO_HOJIN
    splice = SPLICE if splice is None else splice

    w = wage.copy()
    for new_name, (old_name, last_fy) in splice.items():
        old = w[(w["industry"] == old_name) & (w["fy"] <= last_fy)].copy()
        if old.empty:
            continue
        old["industry"] = new_name
        w = pd.concat([w[~((w["industry"] == new_name) & (w["fy"] <= last_fy))],
                       old], ignore_index=True)

    rows = []
    for code, industries in mapping.items():
        sub = w[w["industry"].isin(industries)]
        missing = set(industries) - set(sub["industry"].unique())
        if missing and verbose:
            print(f"  warning: no source industry for nkilm={code}: {missing}")
        if sub.empty:
            continue
        if len(industries) == 1:
            r = sub[["fy", "wage_per_head"]].copy()
        else:
            g = sub.dropna(subset=["n_emp"]).groupby("fy")
            r = (g.apply(lambda x: np.average(x["wage_per_head"],
                                              weights=x["n_emp"]))
                  .rename("wage_per_head").reset_index())
        r["nkilm"] = code
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    return out[["nkilm", "fy", "wage_per_head"]].dropna()


def diagnose_wage(wage_map: pd.DataFrame, fy_min: int = 1995,
                  fy_max: int = 2024, jump_pct: float = 0.15) -> pd.DataFrame:
    """Check the imputed wage series for gaps and discontinuities.

    A classification revision that swaps one source series for another leaves a
    step in the imputed wage.  Year-on-year changes above `jump_pct` are flagged:
    genuine wage growth does not reach that magnitude at the industry level, so
    a flagged year almost always indicates a broken series rather than an
    economic event.
    """
    w = (wage_map[wage_map["fy"].between(fy_min, fy_max)]
         .sort_values(["nkilm", "fy"]))
    w["chg"] = w.groupby("nkilm")["wage_per_head"].pct_change()
    rows = []
    for code, g in w.groupby("nkilm"):
        years = set(g["fy"].astype(int))
        gaps = sorted(set(range(fy_min, fy_max + 1)) - years)
        big = g[g["chg"].abs() > jump_pct]
        rows.append({"nkilm": code, "industry": NKILM_NAME_EN.get(code, "?"),
                     "first": int(g["fy"].min()), "last": int(g["fy"].max()),
                     "missing_years": len(gaps),
                     "jumps": ", ".join(f"{int(r.fy)} ({r.chg:+.0%})"
                                        for r in big.itertuples()) or "-"})
    return pd.DataFrame(rows)


def check_mapping(panel: pd.DataFrame,
                  mapping: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Report how much of the panel the industry mapping covers."""
    mapping = mapping or NKILM_TO_HOJIN
    count = panel.groupby("nkilm")["stkno"].nunique()
    rows = []
    for code in sorted(panel["nkilm"].dropna().unique()):
        rows.append({"nkilm": code, "industry": NKILM_NAME_EN.get(code, "?"),
                     "firms": int(count.get(code, 0)),
                     "obs": int((panel["nkilm"] == code).sum()),
                     "source": " + ".join(mapping.get(code, [])) or "UNMAPPED"})
    out = pd.DataFrame(rows)
    bad = out[out["source"] == "UNMAPPED"]
    print(f"unmapped: {bad['firms'].sum():,} of {out['firms'].sum():,} firms"
          f" ({bad['obs'].sum() / len(panel):.1%} of observations)")
    return out


def attach_value_added(panel: pd.DataFrame, wage_map: pd.DataFrame,
                       fallback_industry: str = FALLBACK_INDUSTRY,
                       wage_all: pd.DataFrame | None = None,
                       theta_k: float = 0.30,
                       gamma_l: float = 0.60) -> pd.DataFrame:
    """Attach imputed labour cost, value added and value-added based measures.

    Industries absent from the mapping fall back to the all-industry average
    when `wage_all` is supplied.

    Columns created
    ---------------
    wage_per_head  compensation per head, millions of yen
    labor_cost     wage_per_head x employees, in the same units as NEEDS
    value_added    operating profit + depreciation + labour cost
    log_z_va       productivity residual on a value added basis
    mrpk_va        log(value added / k_{t-1})
    """
    d = panel.merge(wage_map, on=["nkilm", "fy"], how="left")

    if wage_all is not None:
        fallback = (wage_all[wage_all["industry"] == fallback_industry]
                    [["fy", "wage_per_head"]]
                    .rename(columns={"wage_per_head": "_w_all"}))
        d = d.merge(fallback, on="fy", how="left")
        d["wage_per_head"] = d["wage_per_head"].fillna(d["_w_all"])
        d = d.drop(columns=["_w_all"])

    print(f"  wage data missing for {d['wage_per_head'].isna().mean():.1%} "
          "of observations")

    d["labor_cost"] = d["wage_per_head"] * d["employees"]
    d["value_added"] = (d["op_profit"].fillna(0) + d["depreciation"].fillna(0)
                        + d["labor_cost"])
    d.loc[d["labor_cost"].isna() | d["op_profit"].isna(), "value_added"] = np.nan

    ok = (d["value_added"] > 0) & (d["ppe"] > 0) & (d["employees"] > 0)
    d["log_z_va"] = np.where(
        ok, np.log(d["value_added"]) - theta_k * np.log(d["ppe"])
            - gamma_l * np.log(d["employees"]), np.nan)

    ppe_lag = d.groupby("stkno")["ppe"].shift(1)
    d["mrpk_va"] = np.log((d["value_added"] / ppe_lag).where(
        (d["value_added"] > 0) & (ppe_lag > 0)))
    return d


def validate(df: pd.DataFrame, roa_col: str = "roa") -> None:
    """Check whether the productivity residual behaves as the model implies.

    In theory log(profit/assets) is approximately eta * log z + (theta - 1) *
    log k.  With a sales-based residual this relation failed: an R-squared of
    0.01 and a coefficient on log capital indistinguishable from zero.  The
    diagnostic asks whether moving to a value added basis, which shares the
    denominator and nets out intermediates, restores it.
    """
    import statsmodels.api as sm

    d = df.copy()
    d["_lpi"] = np.log((d["value_added"] / d["ppe"]).where(
        (d["value_added"] > 0) & (d["ppe"] > 0)))
    d["_lk"] = np.log(d["ppe"].where(d["ppe"] > 0))
    for col, label in [("log_z", "sales-based z"),
                       ("log_z_va", "value-added-based z")]:
        if col not in d.columns:
            continue
        s = d.dropna(subset=["_lpi", col, "_lk"])
        if len(s) < 100:
            continue
        m = sm.OLS(s["_lpi"], sm.add_constant(s[[col, "_lk"]])).fit(
            cov_type="cluster", cov_kwds={"groups": s["stkno"]})
        print(f"  [{label}] regression of log(value added / k)  "
              f"R2={m.rsquared:.3f}  n={len(s):,}")
        print(f"      {col}: {m.params[col]:+.4f} ({m.bse[col]:.4f})"
              f"   log k: {m.params['_lk']:+.4f} ({m.bse['_lk']:.4f})")
        print(f"      corr({col}, ROA) = {s[col].corr(s[roa_col]):+.3f}")
