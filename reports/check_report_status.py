#!/usr/bin/env python3
"""reports/ の「歴史記録」と「現在の主張」の分離を機械的に検査する。

`AGENTS.md` の reports 節が定める 2 つの規約を検査する。

1. **身分の明示**: `reports/*.md` はすべて H1 の直後に `Status:` 行を持ち、
   種別が `current` / `record` / `snapshot` / `survey` のいずれかで、
   初出日（ISO）を持つ。
2. **現在値の正本**: 「現在」「現行」「最速」を名乗る行が µs/stage・ms/stage・
   ms/step の数値を書くとき、その数値は `measurement_inventory.md` §18.3 の
   現行表に現れなければならない。当時の値として書くなら「当時」「時点」
   「初出」などの限定語を同じ行に置く。

使い方:  python3 reports/check_report_status.py [--quiet]
終了コード: 0 = 違反なし、1 = ERROR あり（WARN だけなら 0）。
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(HERE, "measurement_inventory.md")

KINDS = ("current", "record", "snapshot", "survey")

STATUS_RE = re.compile(
    r"^>\s*\*\*Status:\*\*\s*`(?P<kind>[a-z]+)`"
    r"（(?:初出\s*)?(?P<date>\d{4}-\d{2}-\d{2})"
)

# 「現在」を名乗る行かどうか
CLAIM_RE = re.compile(r"現在|現行|最速")
# 「当時の値である」と限定している行は対象外
QUALIFIER_RE = re.compile(
    r"当時|時点|初出|追記|訂正|以前|旧|当初|スナップショット|snapshot|"
    r"だった|であった|残す|書き換え"
)
# 数値 + 単位（カンマ・全角空白つきも拾う）
NUM_RE = re.compile(
    r"(?P<num>\d[\d\s,]*\.?\d*)\s*(?P<unit>µs/stage|us/stage|ms/stage|ms/step)"
)
# §18.3 の行:  | 7 | `FUSED` | 1.124 | 292.5 | ...
ROW_RE = re.compile(
    r"^\|\s*(?P<p>\d+)\s*\|\s*\**`(?P<path>[A-Z_0-9]+)`\**\s*\|"
    r"\s*\**(?P<msstep>[\d.]+)\**\s*\|\s*\**(?P<usstage>[\d.]+)\**\s*\|"
)


def load_current_values():
    """§18.3 の表から、現在値として許される数値の集合を作る。"""
    vals = set()
    with open(INVENTORY, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    inside = False
    for line in lines:
        if line.startswith("### 18.3"):
            inside = True
            continue
        if inside and line.startswith("### ") and not line.startswith("### 18.3"):
            break
        if not inside:
            continue
        m = ROW_RE.match(line)
        if m:
            us = float(m.group("usstage"))
            vals.add(us)             # µs/stage
            vals.add(us / 1000.0)    # ms/stage として書かれることがある
            vals.add(float(m.group("msstep")))  # ms/step
    if not vals:
        raise SystemExit("ERROR: measurement_inventory.md §18.3 の表を読めなかった")
    return sorted(vals)


def matches_current(value, current, tol=0.005):
    return any(abs(value - c) <= tol * max(c, 1e-12) for c in current)


def status_kind(lines):
    for line in lines[:8]:
        m = STATUS_RE.match(line)
        if m:
            return m.group("kind")
    return None


def check_status_line(path, lines):
    """H1 の直後 5 行以内に妥当な Status 行があること。"""
    errors = []
    h1 = None
    for i, line in enumerate(lines[:5]):
        if line.startswith("# "):
            h1 = i
            break
    if h1 is None:
        return ["%s: H1 見出しが先頭 5 行に無い" % path]
    window = [l for l in lines[h1 + 1:h1 + 6] if l.strip()]
    for line in window:
        m = STATUS_RE.match(line)
        if m:
            if m.group("kind") not in KINDS:
                errors.append(
                    "%s: Status の種別 `%s` は未定義（%s のいずれか）"
                    % (path, m.group("kind"), " / ".join(KINDS))
                )
            return errors
    errors.append(
        "%s: H1 直後に `> **Status:** \\`kind\\`（初出 YYYY-MM-DD）` 行が無い" % path
    )
    return errors


def check_stale_current(path, lines, current, kind):
    """「現在」を名乗る行の数値が §18.3 と一致すること。

    検査範囲は誤検出を避けて次に限る:

    - **冒頭から 2 つ目の `## ` 見出しまで**、つまり前文と第 1 節。
      読者が「この文書の現在地」と読む場所はそこで、実際に stale-current が
      害になったのもそこである。本文の各節は `record` の定義により当時の
      記録なので対象外。
    - 見出し自身が「当初」「時点」や日付で限定している節、および行自身が
      日付や「当時」「時点」を書いている行は対象外。
    """
    warns = []
    section_qualified = False
    top_headings = 0
    for n, line in enumerate(lines, 1):
        if line.startswith("## "):
            top_headings += 1
            if top_headings >= 2:
                break
            section_qualified = bool(
                QUALIFIER_RE.search(line) or re.search(r"\d{4}-\d{2}-\d{2}", line)
            )
        if section_qualified:
            continue
        if not CLAIM_RE.search(line):
            continue
        # 自分で日付を書いている行は「いつの値か」を名乗っているので対象外
        if QUALIFIER_RE.search(line) or re.search(r"\d{4}-\d{2}-\d{2}", line):
            continue
        for m in NUM_RE.finditer(line):
            raw = m.group("num").replace(",", "").replace(" ", "")
            if not raw or raw.endswith("."):
                continue
            value = float(raw)
            if matches_current(value, current):
                continue
            warns.append(
                "%s:%d: 「現在」を名乗る行の %s %s が §18.3 に無い"
                % (path, n, m.group("num").strip(), m.group("unit"))
            )
    return warns


def check_first_section(path, lines, kind):
    """`record` の第 1 節が無限定の「結論」でないこと。

    `p511` / `p575` / `p767` / `p1023` が壊れた形はこれだった —— 本文は
    追記で更新されているのに、`## 1. 結論` という見出しだけが初回測定の
    ままで、そこが文書の現在地として読まれた。見出しに日付か「当初」が
    あれば読者は誤らない。
    """
    if kind != "record":
        return []
    for line in lines:
        if not line.startswith("## "):
            continue
        title = line[3:]
        if re.search(r"結論|まとめ|要約", title) and not (
            re.search(r"\d{4}-\d{2}-\d{2}", title) or QUALIFIER_RE.search(title)
        ):
            return [
                "%s: 第 1 節 `%s` が無限定の結論見出し。"
                "初回測定の結論なら見出しに日付か「当初」を入れる"
                % (path, line.strip())
            ]
        return []
    return []


def main():
    quiet = "--quiet" in sys.argv
    current = load_current_values()
    errors, warns = [], []
    names = sorted(f for f in os.listdir(HERE) if f.endswith(".md"))
    for name in names:
        with open(os.path.join(HERE, name), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        errors += check_status_line(name, lines)
        kind = status_kind(lines)
        warns += check_first_section(name, lines, kind)
        warns += check_stale_current(name, lines, current, kind)

    for e in errors:
        print("ERROR  " + e)
    for w in warns:
        print("WARN   " + w)
    if not quiet:
        print(
            "\n%d ファイル、ERROR %d 件、WARN %d 件"
            "（WARN は誤検出しうる。当時の値なら行に「当時」「時点」等の"
            "限定語を置く）" % (len(names), len(errors), len(warns))
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
