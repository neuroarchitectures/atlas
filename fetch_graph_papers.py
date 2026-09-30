#!/usr/bin/env python3
"""
Fetch 1-3 additional related arXiv papers for each graph architecture.
Queries arXiv API with 3+ second delays between requests, converts to markdown,
and writes into each references/papers/ directory.

Target: each graph architecture's references/papers/ ends with 2-4 markdown files.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import re
import json
import sys

ROOT = "/Users/xiaming/Workspace/atlas/architectures"
ARXIV_API = "https://export.arxiv.org/api/query"
TARGET_COUNT = 3  # aim for 3 additional papers (1 existing + 3 = 4 total)
RATE_LIMIT = 3.5   # seconds between requests (3+ second delays)

# Architecture -> (list of search queries, list of primary arxiv IDs to exclude)
# Queries are ordered from most specific to most general.
ARCH_QUERIES = {
    "gcn": (
        ['ti:"graph convolutional network" semi-supervised',
         'abs:"graph convolutional" node classification',
         'ti:"spectral" graph convolutional network'],
        ["1609.02907", "1101.5211"]
    ),
    "graphsage": (
        ['ti:"GraphSAGE"',
         'abs:"inductive representation learning" large graphs',
         'abs:"neighborhood aggregation" graph neural network'],
        ["1706.02216", "1710.10568"]
    ),
    "gat": (
        ['ti:"graph attention network"',
         'abs:"graph attention" node classification',
         'abs:"attention" graph neural network'],
        ["1710.10903"]
    ),
    "graph-transformer": (
        ['ti:"graph transformer"',
         'abs:"graph transformer" attention',
         'abs:"transformer" graph structured'],
        ["2012.09699"]
    ),
    "lightgcn": (
        ['ti:"LightGCN"',
         'abs:"light graph convolutional" collaborative filtering',
         'abs:"simplified graph convolutional" recommendation'],
        ["2002.02126"]
    ),
    "netgan": (
        ['ti:"NetGAN"',
         'abs:"NetGAN" graph generation',
         'abs:"graph generative adversarial" random walk'],
        ["1803.00816", "1802.03687"]
    ),
    "graphvae": (
        ['ti:"GraphVAE"',
         'abs:"variational autoencoder" graph generation',
         'abs:"graph generation" variational'],
        ["1611.07308", "1802.03480"]
    ),
    "molgan": (
        ['ti:"MolGAN"',
         'abs:"molecular graph" generative adversarial',
         'abs:"molecule generation" graph neural'],
        ["1805.11973"]
    ),
    "dgi": (
        ['ti:"Deep Graph Infomax"',
         'abs:"deep graph infomax" unsupervised',
         'abs:"mutual information" graph neural network'],
        ["1809.10341", "1808.06670"]
    ),
    "cane": (
        ['ti:"context-aware network embedding"',
         'abs:"CANE" relation modeling',
         'abs:"context-aware" network embedding'],
        ["1603.01718", "1602.03609"]
    ),
    "cognn": (
        ['abs:"cooperative graph neural"',
         'abs:"cooperative graph neural network"',
         'abs:"cooperative" graph neural network learning'],
        ["2310.01267"]
    ),
    "hgemb": (
        ['abs:"heterogeneous graph embedding"',
         'ti:"heterogeneous graph embedding"',
         'abs:"hyperbolic" graph embedding'],
        ["1705.08039"]
    ),
    "metapath2vec": (
        ['ti:"metapath2vec"',
         'abs:"metapath2vec" heterogeneous network',
         'abs:"metapath" heterogeneous network embedding'],
        ["1707.00394"]
    ),
    "struc2vec": (
        ['ti:"struc2vec"',
         'abs:"struc2vec" structural identity',
         'abs:"structural identity" node embedding'],
        ["1704.03124", "1704.03165"]
    ),
    "hin2vec": (
        ['ti:"HIN2Vec"',
         'abs:"HIN2Vec" heterogeneous information network',
         'abs:"heterogeneous information network" meta-path'],
        ["1707.03522"]
    ),
    "grarep": (
        ['ti:"GraRep"',
         'abs:"GraRep" graph representation learning',
         'abs:"global structural" graph representation'],
        ["1506.02236"]
    ),
    "pinsage": (
        ['ti:"PinSage"',
         'abs:"PinSage" graph convolutional recommendation',
         'abs:"web-scale" graph convolutional recommendation'],
        ["1806.01973"]
    ),
    "fastgcn": (
        ['ti:"FastGCN"',
         'abs:"fast graph convolutional" sampling',
         'abs:"sampling" graph convolutional network scalable'],
        ["1801.10247"]
    ),
    "stochastic-gcn": (
        ['ti:"stochastic training" graph convolutional',
         'abs:"GraphSAGE" stochastic training',
         'abs:"stochastic" graph convolutional network'],
        ["1710.10568"]
    ),
    "mhgcn": (
        ['abs:"multiplex heterogeneous graph"',
         'abs:"multiplex heterogeneous" graph neural network',
         'abs:"heterogeneous graph neural network" multiplex'],
        []
    ),
    "uagsl": (
        ['abs:"uncertainty-aware" graph structure learning',
         'abs:"uncertainty aware" graph learning',
         'abs:"graph structure learning" uncertainty'],
        []
    ),
    "pi-gnn": (
        ['abs:"physics-informed graph neural"',
         'ti:"physics-informed" graph neural',
         'abs:"physics informed" graph neural network'],
        ["2412.20962"]
    ),
    "sdge": (
        ['abs:"deep gaussian" graph embedding',
         'abs:"Gaussian" graph embedding unsupervised',
         'abs:"graph embedding" Gaussian distribution'],
        ["1707.03815"]
    ),
    "gde": (
        ['abs:"generative distribution embedding"',
         'abs:"distribution embedding" graph',
         'abs:"generative" graph embedding'],
        []
    ),
    "grrgnn": (
        ['abs:"gated residual recurrent" graph',
         'abs:"gated residual" graph neural network',
         'abs:"recurrent graph neural" traffic'],
        []
    ),
    "khg": (
        ['abs:"knowledge hypergraph"',
         'abs:"knowledge" hypergraph prediction',
         'abs:"hypergraph" knowledge graph'],
        ["1906.00137"]
    ),
    "shg": (
        ['abs:"semantic hypergraph"',
         'abs:"semantic" hypergraph',
         'abs:"hypergraph" semantic representation'],
        []
    ),
    "structuralseq-gnn": (
        ['abs:"structural sequence" graph neural',
         'abs:"structural attention" graph classification',
         'abs:"graph classification" structural'],
        []
    ),
    "st-gdn": (
        ['abs:"spatiotemporal graph deviation"',
         'abs:"spatiotemporal" graph deviation network',
         'abs:"spatiotemporal" graph neural network traffic'],
        []
    ),
    "st-mgcn": (
        ['abs:"spatiotemporal multi-graph convolutional"',
         'abs:"spatiotemporal" graph convolutional network traffic',
         'abs:"spatiotemporal" multi-graph traffic prediction'],
        []
    ),
    "gmn": (
        ['ti:"Graph Matching Network"',
         'abs:"graph matching network"',
         'abs:"graph matching" neural network'],
        ["1903.03543"]
    ),
}


def query_arxiv(search_query, max_results=10, retries=3):
    """Query arXiv API and return parsed entries."""
    params = {
        'search_query': search_query,
        'start': '0',
        'max_results': str(max_results),
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AtlasPaperFetcher/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            return parse_arxiv_response(data)
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(10)  # longer backoff on error
            else:
                print(f"    [ERROR] query failed after {retries+1} attempts: {last_err}", flush=True)
    return []


def parse_arxiv_response(xml_text):
    """Parse arXiv Atom XML response into list of dicts."""
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall('atom:entry', ns):
        entry_id = entry.find('atom:id', ns)
        arxiv_url = entry_id.text.strip() if entry_id is not None else ""
        arxiv_id_raw = arxiv_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        base_id = arxiv_id_raw.split('v')[0]

        title_el = entry.find('atom:title', ns)
        title = title_el.text.strip().replace('\n', ' ') if title_el is not None else ""
        title = ' '.join(title.split())

        summary_el = entry.find('atom:summary', ns)
        summary = summary_el.text.strip() if summary_el is not None else ""
        summary = ' '.join(summary.split())

        published_el = entry.find('atom:published', ns)
        published = published_el.text.strip() if published_el is not None else ""

        authors = []
        for author in entry.findall('atom:author', ns):
            name_el = author.find('atom:name', ns)
            if name_el is not None:
                authors.append(name_el.text.strip())

        pdf_url = ""
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break

        categories = []
        for cat in entry.findall('atom:category', ns):
            categories.append(cat.get('term', ''))

        comment_el = entry.find('arxiv:comment', ns)
        comment = comment_el.text.strip() if comment_el is not None else ""

        entries.append({
            'arxiv_id': base_id,
            'arxiv_url': arxiv_url,
            'title': title,
            'summary': summary,
            'published': published,
            'authors': authors,
            'pdf_url': pdf_url,
            'categories': categories,
            'comment': comment,
        })
    return entries


def slugify(s, maxlen=60):
    """Create a filename-safe slug from a string."""
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:maxlen]


def author_short(authors):
    """Short author string for filename: FirstAuthor_et_al."""
    if not authors:
        return "Unknown"
    first = authors[0]
    first = re.sub(r'[^\w\s-]', '', first).strip().replace(' ', '_')
    if len(authors) > 1:
        return f"{first}_et_al."
    return first


def year_from_published(published):
    if published and len(published) >= 4:
        return published[:4]
    return "Unknown"


def build_markdown(entry):
    """Build markdown content for a paper entry."""
    title = entry['title']
    arxiv_url = entry['arxiv_url']
    authors = entry['authors']
    summary = entry['summary']
    published = entry['published']
    year = year_from_published(published)
    categories = entry['categories']
    comment = entry['comment']

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> Source: `{arxiv_url}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    if authors:
        lines.append("**Authors:** " + ", ".join(authors))
        lines.append("")

    if year != "Unknown":
        lines.append(f"**Published:** {published}")
        lines.append("")

    if categories:
        lines.append(f"**Categories:** {', '.join(categories)}")
        lines.append("")

    if comment:
        lines.append(f"**Comments:** {comment}")
        lines.append("")

    lines.append("## Abstract")
    lines.append("")
    lines.append(summary)
    lines.append("")

    if entry['pdf_url']:
        lines.append(f"**PDF:** {entry['pdf_url']}")
        lines.append("")

    return "\n".join(lines)


def get_existing_files(papers_dir):
    """Get set of existing markdown filenames and existing arxiv IDs in a papers dir."""
    existing_files = set()
    existing_ids = set()
    if os.path.isdir(papers_dir):
        for f in os.listdir(papers_dir):
            if f.endswith('.md'):
                existing_files.add(f)
                m = re.search(r'(\d{4}\.\d{4,5})', f)
                if m:
                    existing_ids.add(m.group(1))
    return existing_files, existing_ids


def select_papers(arch, queries, exclude_ids, existing_ids, target=3):
    """Query arXiv and select related papers, excluding primary and existing."""
    selected = []
    seen_ids = set(exclude_ids) | set(existing_ids)

    for q in queries:
        if len(selected) >= target:
            break
        entries = query_arxiv(q, max_results=10)
        time.sleep(RATE_LIMIT)  # 3+ second delay between requests

        for e in entries:
            if len(selected) >= target:
                break
            eid = e['arxiv_id']
            # Skip if already selected, excluded (primary), or existing in dir
            if eid in seen_ids:
                continue
            # Skip entries with empty title or summary
            if not e['title'] or not e['summary']:
                continue
            # Skip obviously irrelevant entries
            skip_titles = ['ILC Technology Network', 'erratum', 'Erratum']
            if any(st in e['title'] for st in skip_titles):
                continue
            seen_ids.add(eid)
            selected.append(e)

    return selected


def main():
    results = {}
    arch_list = sorted(ARCH_QUERIES.keys())

    for i, arch in enumerate(arch_list):
        queries, exclude_ids = ARCH_QUERIES[arch]
        papers_dir = os.path.join(ROOT, arch, "references", "papers")
        existing_files, existing_ids = get_existing_files(papers_dir)

        print(f"[{i+1}/{len(arch_list)}] {arch} (existing: {len(existing_files)} files, {len(existing_ids)} IDs)...", flush=True)

        selected = select_papers(arch, queries, exclude_ids, existing_ids, target=TARGET_COUNT)

        written = 0
        for e in selected:
            year = year_from_published(e['published'])
            author_s = author_short(e['authors'])
            slug = slugify(e['title'], maxlen=50)
            filename = f"{author_s}_{year}_{e['arxiv_id']}.md"
            filepath = os.path.join(papers_dir, filename)

            # Avoid overwriting existing files
            if os.path.exists(filepath):
                continue

            md = build_markdown(e)
            os.makedirs(papers_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(md)
            written += 1
            print(f"  -> wrote {filename}", flush=True)

        results[arch] = written
        print(f"  -> total written: {written}", flush=True)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    total_written = sum(results.values())
    print(f"Total papers written: {total_written}", flush=True)
    zero = [a for a, c in results.items() if c == 0]
    if zero:
        print(f"Architectures with 0 new papers: {zero}", flush=True)

    # Save results for inspection
    with open("/tmp/fetch_graph_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
