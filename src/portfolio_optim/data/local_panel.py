from __future__ import annotations

from pathlib import Path

import pandas as pd

# Typical Stooq ASCII daily file (from https://stooq.com/db/h/) has a header row with
# Date,Open,High,Low,Close,Volume — sometimes column names vary by region.


def _find_col(df: pd.DataFrame, want: str) -> str | None:
    w = want.lower()
    for c in df.columns:
        if str(c).strip().lower() == w:
            return c
    return None


def _clean_colname(c: str) -> str:
    s = str(c).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    return s.strip()


def _read_stooq_like_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", skipinitialspace=True, engine="python", on_bad_lines="skip")
    df.columns = [_clean_colname(c) for c in df.columns]
    return df


def _parse_stooq_dates(series: pd.Series) -> pd.DatetimeIndex:
    """
    Stooq bulk exports often encode dates as YYYYMMDD (e.g. 19970516) under DATE.
    Some other files may already have ISO dates under Date.
    """
    s = series.astype(str).str.strip()
    # Prefer YYYYMMDD if it looks like it.
    is_yyyymmdd = s.str.fullmatch(r"\d{8}", na=False)
    out = pd.to_datetime(s.where(is_yyyymmdd, s), format="%Y%m%d", errors="coerce")
    # For non-YYYYMMDD rows, the format arg above will coerce; retry generic parse.
    need_retry = out.isna() & (~is_yyyymmdd)
    if need_retry.any():
        out2 = pd.to_datetime(s[need_retry], errors="coerce")
        out.loc[need_retry] = out2
    return pd.DatetimeIndex(out)


def load_close_series_from_stooq_ascii_file(path: Path | str) -> pd.Series:
    """
    Load one symbol’s **Close** from a single Stooq-style daily file (CSV-like, often ``.txt``).

    The **series name** is the filename stem (e.g. ``spy.us`` from ``spy.us.txt``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    df = _read_stooq_like_csv(path)

    # Supports both:
    # - Stooq db/h per-symbol files: Date,Open,...,Close,...
    # - Stooq bulk-like files: TICKER,PER,DATE,TIME,OPEN,...,CLOSE,...
    c_date = _find_col(df, "Date") or _find_col(df, "DATE") or _find_col(df, "date")
    c_close = _find_col(df, "Close") or _find_col(df, "CLOSE") or _find_col(df, "close")
    if c_date is None or c_close is None:
        raise ValueError(f"Unrecognized Stooq file schema in {path!r}; columns={list(df.columns)}")

    dt = _parse_stooq_dates(df[c_date])
    close = pd.to_numeric(df[c_close], errors="coerce")

    # Name the series: prefer TICKER if present, otherwise filename stem
    c_ticker = _find_col(df, "TICKER") or _find_col(df, "ticker")
    if c_ticker is not None:
        uniq = pd.Series(df[c_ticker].astype(str).str.strip().unique())
        uniq = uniq[uniq != ""]
        name = str(uniq.iloc[0]) if len(uniq) >= 1 else path.stem
        if len(uniq) > 1:
            # This helper returns a single Series; multi-ticker files are handled by the directory loader.
            df = df[df[c_ticker].astype(str).str.strip() == name]
            dt = _parse_stooq_dates(df[c_date])
            close = pd.to_numeric(df[c_close], errors="coerce")
    else:
        name = path.stem

    out = pd.Series(close.to_numpy(), index=dt, name=name)
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    if out.empty:
        raise ValueError(f"No usable rows in {path!r}")
    return out


def load_close_panel_from_stooq_dir(
    directory: Path | str,
    *,
    glob_pattern: str = "*.txt",
) -> pd.DataFrame:
    """
    Build a wide **Close** panel from a folder of per-symbol Stooq ASCII files (e.g. after
    unzipping a download from https://stooq.com/db/h/).

    Uses an **inner** join on dates (only dates present for every symbol).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(directory.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} under {directory}")

    cols: list[pd.Series] = []
    for p in paths:
        if p.is_dir():
            continue
        # If a file contains multiple tickers, split it; otherwise treat as one series.
        df = _read_stooq_like_csv(p)
        c_ticker = _find_col(df, "TICKER") or _find_col(df, "ticker")
        if c_ticker is None:
            cols.append(load_close_series_from_stooq_ascii_file(p))
            continue

        tickers = [t for t in df[c_ticker].astype(str).str.strip().unique() if t]
        if not tickers:
            cols.append(load_close_series_from_stooq_ascii_file(p))
            continue

        # Write series per ticker in this file.
        c_date = _find_col(df, "Date") or _find_col(df, "DATE") or _find_col(df, "date")
        c_close = _find_col(df, "Close") or _find_col(df, "CLOSE") or _find_col(df, "close")
        if c_date is None or c_close is None:
            raise ValueError(f"Unrecognized Stooq file schema in {p!r}; columns={list(df.columns)}")
        for t in tickers:
            sub = df[df[c_ticker].astype(str).str.strip() == t]
            dt = _parse_stooq_dates(sub[c_date])
            close = pd.to_numeric(sub[c_close], errors="coerce")
            s = pd.Series(close.to_numpy(), index=dt, name=str(t))
            s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
            if not s.empty:
                cols.append(s)

    panel = pd.concat(cols, axis=1)
    panel = panel.sort_index().dropna(how="any")
    return panel


def download_returns_from_stooq_local_dir(
    directory: Path | str,
    *,
    glob_pattern: str = "*.txt",
) -> pd.DataFrame:
    """Simple daily returns from ``pct_change`` on the close panel."""
    close = load_close_panel_from_stooq_dir(directory, glob_pattern=glob_pattern)
    return close.pct_change().dropna(how="any")

def download_returns_from_stooq_local_dir_raw(
            directory: Path | str,
    *,
    glob_pattern: str = "*.txt",
) -> pd.DataFrame:
    """Simple daily returns from ``pct_change`` on the close panel."""
    close = load_close_panel_from_stooq_dir(directory, glob_pattern=glob_pattern)
    return close



def load_returns_from_csv_wide(
    path: Path | str,
    *,
    date_col: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a **wide** CSV you maintain by hand: one column per asset + a date column.

    Column names become asset identifiers; first column or ``date_col`` must be dates.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Expected column {date_col!r}; got {list(df.columns)}")
    out = df.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(df[date_col], errors="coerce")
    out.index = idx
    out = out.sort_index().dropna(how="all", axis=0)
    out.index.name = "date"
    return out.pct_change().dropna(how="any")
