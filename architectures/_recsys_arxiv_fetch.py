#!/usr/bin/env python3
"""
Query arXiv API for related papers for recsys architectures that currently
have only 1 paper in references/papers/. Fetch 1-3 additional papers each,
with strict rate-limit backoff (5+ second delays, exponential backoff on 429).
"""

import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import http.client
import socket
import sys

BASE = "/Users/xiaming/Workspace/atlas/architectures"
LOG_FILE = "/tmp/recsys_fetch_log.txt"

def log(msg):
    """Write to both stdout and a log file."""
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
        f.flush()

# Only architectures that currently have exactly 1 paper and need enrichment
# (ncf=7, neumf=4, deepfm=2, din=2, two-tower=2 already have enough)
ARCH_QUERIES = {
    "wide-and-deep": [
        'ti:"wide and deep" AND abs:recommendation',
        'ti:"wide deep" AND abs:memorization AND abs:generalization',
        'ti:"deep neural" AND ti:"wide" AND abs:recommendation AND abs:click',
    ],
    "dien": [
        'ti:"deep interest evolving network"',
        'ti:"interest evolving" AND abs:recommendation',
        'ti:"sequential" AND ti:"interest" AND abs:click-through rate',
    ],
    "bst": [
        'ti:"behavior sequence transformer" AND abs:recommendation',
        'ti:"transformer" AND ti:"user behavior" AND abs:recommendation',
        'ti:"sequential" AND ti:"transformer" AND abs:recommendation AND abs:click',
    ],
    "gru4rec": [
        'ti:"session-based" AND ti:"recurrent" AND abs:recommendation',
        'ti:"GRU4Rec"',
        'ti:"session-based" AND abs:neural AND abs:recommendation',
    ],
    "sasrec": [
        'ti:"self-attentive" AND ti:"sequential" AND abs:recommendation',
        'ti:"SASRec" AND abs:recommendation',
        'ti:"transformer" AND ti:"sequential recommendation"',
    ],
    "mmoe": [
        'ti:"mixture of experts" AND abs:multi-task AND abs:recommendation',
        'ti:"multi-gate" AND ti:"mixture" AND abs:recommendation',
        'ti:"multi-task" AND ti:"expert" AND abs:recommendation AND abs:neural',
    ],
    "ple": [
        'ti:"progressive layered extraction" AND abs:recommendation',
        'ti:"PLE" AND abs:multi-task AND abs:recommendation',
        'ti:"customized gate control" AND abs:multi-task',
    ],
    "esmm": [
        'ti:"entire space" AND ti:"multi-task" AND abs:recommendation',
        'ti:"ESMM" AND abs:conversion',
        'ti:"post-click" AND abs:recommendation AND abs:conversion',
    ],
    "dlrm": [
        'ti:"deep learning recommendation model"',
        'ti:"DLRM" AND abs:recommendation',
        'ti:"embedding" AND ti:"multilayer perceptron" AND abs:recommendation',
    ],
    "masa": [
        'ti:"matching anything" AND abs:segmentation',
        'ti:"segment anything" AND abs:matching',
        'ti:"universal" AND ti:"matching" AND abs:segmentation',
    ],
    "harp": [
        'ti:"hierarchical" AND ti:"representation" AND abs:network embedding',
        'ti:"graph" AND ti:"embedding" AND abs:hierarchical AND abs:compression',
        'ti:"network embedding" AND abs:hierarchical AND abs:coarsening',
    ],
    "gde": [
        'ti:"generative distribution embeddings"',
        'ti:"distribution embedding" AND abs:generative',
        'ti:"node embedding" AND abs:distribution AND abs:generative',
    ],
    "msde": [
        'ti:"multiscale" AND ti:"dynamical" AND abs:embedding',
        'ti:"multi-scale" AND ti:"embedding" AND abs:network',
        'ti:"dynamical" AND ti:"embedding" AND abs:complex networks',
    ],
}


def query_arxiv(query, max_results=5, max_retries=6):
    """Query arXiv API with exponential backoff on HTTP 429."""
    search_url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}"
    )
    req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0 (research-paper-fetcher)'})

    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(req, timeout=45)
            xml_data = response.read().decode('utf-8')
            return parse_arxiv_response(xml_data)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 5, 160)  # 5, 10, 20, 40, 80, 160s
                print(f"    HTTP 429 (rate limited). Backing off {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                # Rebuild request (connection may be closed)
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0 (research-paper-fetcher)'})
                continue
            else:
                print(f"    HTTP error {e.code}: {e.reason}")
                return []
        except (urllib.error.URLError, socket.timeout, http.client.HTTPException) as e:
            wait = min(2 ** attempt * 5, 160)
            print(f"    Network error: {e}. Retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0 (research-paper-fetcher)'})
            continue

    print(f"    Max retries exceeded for query: {query}")
    return []


def parse_arxiv_response(xml_data):
    """Parse arXiv Atom XML response into list of paper dicts."""
    papers = []
    try:
        root = ET.fromstring(xml_data)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        for entry in root.findall('atom:entry', ns):
            entry_id = entry.find('atom:id', ns)
            if entry_id is None:
                continue
            arxiv_url = entry_id.text.strip()
            id_match = re.search(r'abs/(\d+\.\d+)', arxiv_url)
            if not id_match:
                continue
            arxiv_id = id_match.group(1)

            title_el = entry.find('atom:title', ns)
            title = title_el.text.strip() if title_el is not None and title_el.text else ''
            title = re.sub(r'\s+', ' ', title)

            summary_el = entry.find('atom:summary', ns)
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ''
            summary = re.sub(r'\s+', ' ', summary)

            authors = []
            for author in entry.findall('atom:author', ns):
                name_el = author.find('atom:name', ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            pub_el = entry.find('atom:published', ns)
            published = pub_el.text.strip() if pub_el is not None and pub_el.text else ''

            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term', '')
                if term:
                    categories.append(term)

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            papers.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': summary,
                'authors': authors,
                'published': published,
                'categories': categories,
                'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                'pdf_url': pdf_url,
            })
    except ET.ParseError as e:
        print(f"    XML parse error: {e}")
    return papers


def get_existing_arxiv_ids(arch):
    """Get set of arxiv IDs already in the papers directory.
    Only look at the 'Source:' line to avoid matching citation references."""
    papers_path = os.path.join(BASE, arch, "references", "papers")
    ids = set()
    if not os.path.isdir(papers_path):
        return ids
    for fname in os.listdir(papers_path):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(papers_path, fname)
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    # Only match the Source line (first few lines)
                    if 'Source:' in line:
                        matches = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', line)
                        ids.update(matches)
                        break
            # Also extract from filename
            fn_matches = re.findall(r'(\d{4}\.\d+)', fname)
            ids.update(fn_matches)
        except:
            pass
    return ids


def sanitize_filename(title, arxiv_id):
    """Create a clean filename from paper title and arxiv ID."""
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())[:80]
    return f"{clean}_{arxiv_id}.md"


def create_paper_md(paper, arch):
    """Create markdown content for a paper."""
    authors_str = ", ".join(paper['authors'][:10])
    if len(paper['authors']) > 10:
        authors_str += " et al."

    year = ""
    if paper['published']:
        year = paper['published'][:4]

    first_author = paper['authors'][0].split()[-1] if paper['authors'] else "Unknown"
    label = f"{first_author} et al. {year}" if paper['authors'] else f"Unknown {year}"

    md = f"""# Paper ({label})

> Source: `https://arxiv.org/abs/{paper['arxiv_id']}`

---

**{paper['title']}**

{authors_str}

**Abstract**

{paper['summary']}

---

- **arXiv ID**: `{paper['arxiv_id']}`
- **Published**: {paper['published'][:10] if paper['published'] else 'N/A'}
- **Categories**: {', '.join(paper['categories']) if paper['categories'] else 'N/A'}
- **PDF**: {paper['pdf_url']}
"""
    return md


def is_relevant(paper, arch):
    """Basic relevance check to filter out obviously off-topic papers."""
    title_lower = paper['title'].lower()
    summary_lower = paper['summary'].lower() if paper['summary'] else ''

    # For CRISPR/biology papers that share the name but are unrelated
    if 'crispr' in title_lower or 'cas9' in title_lower or 'cas13' in title_lower:
        return False
    # For gene/genomics unrelated papers
    if 'gene editing' in title_lower and 'recommendation' not in summary_lower:
        return False

    return True


def process_architecture(arch, queries, target_count=3):
    """Query arXiv for related papers and write markdown files."""
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        os.makedirs(papers_path, exist_ok=True)

    existing_ids = get_existing_arxiv_ids(arch)
    print(f"  {arch}: existing IDs = {existing_ids}")

    collected = []
    seen_ids = set(existing_ids)

    for query in queries:
        if len(collected) >= target_count:
            break
        print(f"    Query: {query}", flush=True)
        results = query_arxiv(query, max_results=5)
        # Strict rate limit: 5+ second delays between requests
        time.sleep(6)

        for paper in results:
            if len(collected) >= target_count:
                break
            if paper['arxiv_id'] in seen_ids:
                continue
            if len(paper['title']) < 10:
                continue
            if not is_relevant(paper, arch):
                print(f"    Skipping irrelevant: {paper['arxiv_id']} - {paper['title'][:60]}")
                continue
            seen_ids.add(paper['arxiv_id'])
            collected.append(paper)
            print(f"    Found: {paper['arxiv_id']} - {paper['title'][:70]}", flush=True)

    # Write markdown files
    written = 0
    for paper in collected:
        md_content = create_paper_md(paper, arch)
        filename = sanitize_filename(paper['title'], paper['arxiv_id'])
        fpath = os.path.join(papers_path, filename)
        if os.path.exists(fpath):
            filename = f"Related_{paper['arxiv_id']}.md"
            fpath = os.path.join(papers_path, filename)
        with open(fpath, 'w') as f:
            f.write(md_content)
        written += 1
        print(f"    Written: {filename}", flush=True)

    return written


def main():
    total_written = 0
    for arch, queries in ARCH_QUERIES.items():
        print(f"\n=== Processing: {arch} ===", flush=True)
        written = process_architecture(arch, queries, target_count=3)
        total_written += written
        print(f"  {arch}: {written} papers written", flush=True)
        # Extra delay between architectures
        time.sleep(3)

    print(f"\n=== DONE: {total_written} total papers written ===", flush=True)


if __name__ == '__main__':
    main()
