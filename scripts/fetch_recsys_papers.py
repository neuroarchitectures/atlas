#!/usr/bin/env python3
"""
Fetch 1-3 additional related arXiv papers for each recsys architecture and
write markdown files into each architecture's references/papers/ directory.

Respects arXiv rate limits: 3.5s sleep between every API request, with
exponential backoff on errors.

Target: each recsys architecture ends up with 2-4 markdown paper files
(primary already present + 1-3 new related papers).
"""
import os
import re
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

BASE = "/Users/xiaming/Workspace/atlas/architectures"
ARXIV_API = "https://export.arxiv.org/api/query"
SLEEP_BETWEEN = 3.5          # >3s between requests (rate-limit backoff)
TARGET_NEW = 3               # aim for 3 new papers (primary already present -> 2-4 total)
MAX_RESULTS = 8              # results to pull per query

# Architecture -> (search queries, primary arxiv IDs to exclude)
# Queries are crafted to surface closely related work, not the primary paper.
ARCH_QUERIES = {
    "ncf": (
        [
            'ti:"neural collaborative filtering"',
            'abs:"neural collaborative filtering" recommendation',
            'abs:"collaborative filtering" deep learning implicit feedback',
        ],
        ["1708.05031"],
    ),
    "neumf": (
        [
            'ti:"neural matrix factorization"',
            'abs:"neural matrix factorization" recommendation',
            'abs:"matrix factorization" neural network recommendation',
        ],
        ["1708.05031"],
    ),
    "deepfm": (
        [
            'ti:"DeepFM"',
            'abs:"factorization machine" deep learning CTR',
            'abs:"deep factorization machine" click-through',
        ],
        ["1703.04247"],
    ),
    "wide-and-deep": (
        [
            'ti:"Wide and Deep"',
            'abs:"wide and deep learning" recommender',
            'abs:"wide and deep" CTR prediction',
        ],
        ["1606.07792"],
    ),
    "din": (
        [
            'ti:"Deep Interest Network"',
            'abs:"deep interest network" CTR',
            'abs:"attentional factorization" recommendation',
        ],
        ["1706.06978", "1703.06211"],
    ),
    "dien": (
        [
            'ti:"Deep Interest Evolution Network"',
            'abs:"deep interest evolution network"',
            'abs:"interest evolving" sequential recommendation',
        ],
        ["1809.03672"],
    ),
    "bst": (
        [
            'ti:"Behavior Sequence Transformer"',
            'abs:"behavior sequence transformer" recommendation',
            'abs:"transformer" user behavior recommendation CTR',
        ],
        ["1905.06874"],
    ),
    "gru4rec": (
        [
            'ti:"session-based recommendations" recurrent',
            'abs:"session-based" GRU recommendation',
            'abs:"recurrent neural network" session recommendation',
        ],
        ["1511.06939"],
    ),
    "sasrec": (
        [
            'ti:"Self-Attentive Sequential Recommendation"',
            'abs:"self-attentive" sequential recommendation',
            'abs:"transformer" sequential recommendation attention',
        ],
        ["1808.09781"],
    ),
    "mmoe": (
        [
            'ti:"Multi-Gate Mixture-of-Experts"',
            'abs:"mixture of experts" multi-task learning',
            'abs:"multi-gate" mixture experts recommendation',
        ],
        ["1705.02343"],
    ),
    "ple": (
        [
            'ti:"Progressive Layered Extraction"',
            'abs:"progressive layered extraction" multi-task',
            'abs:"multi-task learning" shared experts recommendation',
        ],
        ["2005.08100"],
    ),
    "esmm": (
        [
            'ti:"Entire Space Multi-Task Model"',
            'abs:"entire space" multi-task model',
            'abs:"conversion rate" multi-task entire space',
        ],
        ["1804.07931"],
    ),
    "two-tower": (
        [
            'abs:"two-tower" neural network recommendation',
            'abs:"dual tower" retrieval model',
            'abs:"two tower" embedding retrieval',
        ],
        ["1906.03309"],
    ),
    "dlrm": (
        [
            'ti:"Deep Learning Recommendation Model"',
            'abs:"deep learning recommendation model" DLRM',
            'abs:"recommendation model" deep learning production',
        ],
        ["1906.00091"],
    ),
    "masa": (
        [
            'abs:"memory augmented" sequential recommendation',
            'abs:"memory augmented" session recommendation',
            'abs:"MASA" recommendation',
        ],
        ["2406.04221"],
    ),
    "harp": (
        [
            'abs:"hierarchical representation learning" networks',
            'abs:"hierarchical" network embedding node',
            'abs:"HARP" network representation',
        ],
        ["1606.07792"],
    ),
    "gde": (
        [
            'abs:"graph dynamical embedding"',
            'abs:"continuous-time dynamic graph" embedding',
            'abs:"dynamic graph" representation learning temporal',
        ],
        [],
    ),
    "msde": (
        [
            'abs:"multi-scale" dynamic embedding',
            'abs:"multiscale dynamical embeddings" networks',
            'abs:"multi-scale" network embedding temporal',
        ],
        [],
    ),
}


def query_arxiv(query, max_results=MAX_RESULTS, retries=3):
    """Query arXiv API with exponential backoff on failure."""
    params = {
        'search_query': query,
        'start': '0',
        'max_results': str(max_results),
        'sortBy': 'relevance',
        'sortOrder': 'descending',
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'AtlasRecsysPaperFetcher/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            return parse_arxiv_response(data)
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                wait = 3.5 * (2 ** attempt)  # 3.5, 7, 14s backoff
                print(f"    retry {attempt+1}/{retries} after {wait}s: {last_err}")
                time.sleep(wait)
    print(f"    FAILED query after retries: {last_err}")
    return []


def parse_arxiv_response(xml_text):
    """Parse arXiv Atom XML into list of paper dicts."""
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
    }
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall('atom:entry', ns):
        entry_id = entry.find('atom:id', ns)
        if entry_id is None or not entry_id.text:
            continue
        arxiv_url = entry_id.text.strip()
        m = re.search(r'abs/(\d+\.\d+)', arxiv_url)
        if not m:
            continue
        arxiv_id = m.group(1)

        title_el = entry.find('atom:title', ns)
        title = ''
        if title_el is not None and title_el.text:
            title = re.sub(r'\s+', ' ', title_el.text.strip())

        summary_el = entry.find('atom:summary', ns)
        summary = ''
        if summary_el is not None and summary_el.text:
            summary = re.sub(r'\s+', ' ', summary_el.text.strip())

        authors = []
        for author in entry.findall('atom:author', ns):
            name_el = author.find('atom:name', ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        published_el = entry.find('atom:published', ns)
        published = published_el.text.strip() if published_el is not None and published_el.text else ''

        categories = []
        for cat in entry.findall('atom:category', ns):
            term = cat.get('term', '')
            if term:
                categories.append(term)

        comment_el = entry.find('arxiv:comment', ns)
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ''

        entries.append({
            'arxiv_id': arxiv_id,
            'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}",
            'title': title,
            'summary': summary,
            'published': published,
            'authors': authors,
            'categories': categories,
            'comment': comment,
        })
    return entries


def get_existing_ids(papers_dir):
    """Get set of arxiv IDs already present (from filenames + file contents)."""
    ids = set()
    if not os.path.isdir(papers_dir):
        return ids, 0
    count = 0
    for fname in os.listdir(papers_dir):
        if not fname.endswith('.md'):
            continue
        count += 1
        for m in re.findall(r'(\d{4}\.\d{4,5})', fname):
            ids.add(m)
        try:
            with open(os.path.join(papers_dir, fname), 'r') as f:
                content = f.read()
            for m in re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content):
                ids.add(m)
        except Exception:
            pass
    return ids, count


def sanitize_filename(title, arxiv_id):
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())[:80]
    clean = clean.strip('_')
    if not clean:
        clean = "Related"
    return f"{clean}_{arxiv_id}.md"


def build_markdown(paper):
    """Build markdown matching the repo's existing arXiv-paper format."""
    authors_str = ", ".join(paper['authors'][:10])
    if len(paper['authors']) > 10:
        authors_str += " et al."

    year = paper['published'][:4] if paper['published'] else "Unknown"
    first_author = paper['authors'][0].split()[-1] if paper['authors'] else "Unknown"
    label = f"{first_author} et al. {year}" if paper['authors'] else f"Unknown {year}"

    pub_date = paper['published'][:10] if paper['published'] else 'N/A'
    cats = ', '.join(paper['categories']) if paper['categories'] else 'N/A'

    md = f"""# Paper ({label})

> Source: `https://arxiv.org/abs/{paper['arxiv_id']}`

---

**{paper['title']}**

{authors_str}

**Abstract**

{paper['summary']}

---

- **arXiv ID**: `{paper['arxiv_id']}`
- **Published**: {pub_date}
- **Categories**: {cats}
- **PDF**: {paper['pdf_url']}
"""
    return md


def select_papers(queries, exclude_ids, existing_ids, target=TARGET_NEW):
    """Query arXiv and select related papers, excluding primary + existing."""
    selected = []
    seen = set(exclude_ids) | set(existing_ids)

    for q in queries:
        if len(selected) >= target:
            break
        entries = query_arxiv(q, max_results=MAX_RESULTS)
        time.sleep(SLEEP_BETWEEN)  # rate-limit backoff between requests

        for e in entries:
            if len(selected) >= target:
                break
            eid = e['arxiv_id']
            if eid in seen:
                continue
            if not e['title'] or not e['summary']:
                continue
            if len(e['title']) < 10:
                continue
            seen.add(eid)
            selected.append(e)
            print(f"    found: {eid} - {e['title'][:70]}")

    return selected


def main():
    results = {}
    archs = sorted(ARCH_QUERIES.keys())

    for i, arch in enumerate(archs):
        queries, exclude_ids = ARCH_QUERIES[arch]
        papers_dir = os.path.join(BASE, arch, "references", "papers")
        existing_ids, existing_count = get_existing_ids(papers_dir)

        print(f"\n[{i+1}/{len(archs)}] {arch} (existing: {existing_count} files)...")

        selected = select_papers(queries, exclude_ids, existing_ids, target=TARGET_NEW)

        written = 0
        for e in selected:
            filename = sanitize_filename(e['title'], e['arxiv_id'])
            fpath = os.path.join(papers_dir, filename)
            if os.path.exists(fpath):
                filename = f"Related_{e['arxiv_id']}.md"
                fpath = os.path.join(papers_dir, filename)
            os.makedirs(papers_dir, exist_ok=True)
            with open(fpath, 'w') as f:
                f.write(build_markdown(e))
            written += 1
            print(f"    wrote: {filename}")

        results[arch] = {"new": written, "total": existing_count + written}
        print(f"  -> {arch}: +{written} new, total now {existing_count + written}")

    print("\n=== SUMMARY ===")
    total_new = sum(r["new"] for r in results.values())
    print(f"Total new papers written: {total_new}")
    for arch in archs:
        r = results[arch]
        flag = "" if 2 <= r["total"] <= 4 else "  <-- OUT OF RANGE"
        print(f"  {arch:14s}: {r['new']} new, {r['total']} total{flag}")

    with open("/tmp/recsys_fetch_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
