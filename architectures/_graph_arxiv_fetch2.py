#!/usr/bin/env python3
"""Fetch additional arXiv papers for graph architectures that still need them."""
import os, re, time, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

BASE = "/Users/xiaming/Workspace/atlas/architectures"
BASE_DELAY = 10.0
MAX_RETRIES = 4

ARCH_QUERIES = {
    "cognn": [
        'ti:"cooperative" AND ti:"graph neural"',
        'ti:"CoGNN" OR ti:"cooperative graph"',
        'ti:"graph neural network" AND abs:cooperative',
        'ti:"multi-agent" AND abs:graph neural',
    ],
    "metapath2vec": [
        'ti:"metapath2vec" OR ti:"meta-path" AND abs:embedding',
        'ti:"heterogeneous" AND ti:"network" AND abs:embedding',
        'ti:"meta-path" AND abs:"representation learning"',
        'ti:"heterogeneous" AND ti:"graph" AND abs:embedding',
    ],
    "struc2vec": [
        'ti:"struc2vec" OR ti:"structural identity" AND abs:embedding',
        'ti:"structural" AND ti:"node" AND abs:embedding',
        'ti:"structural similarity" AND abs:graph AND abs:embedding',
        'ti:"structural" AND ti:"representation" AND abs:graph',
    ],
    "hin2vec": [
        'ti:"HIN2Vec" OR ti:"heterogeneous information network"',
        'ti:"meta-path" AND ti:"heterogeneous" AND abs:embedding',
        'ti:"heterogeneous" AND ti:"network" AND abs:"representation learning"',
        'ti:"heterogeneous" AND abs:"information network" AND abs:embedding',
    ],
    "grarep": [
        'ti:"GraRep" OR ti:"global structural" AND abs:graph',
        'ti:"graph representation" AND ti:"global" AND abs:structural',
        'ti:"transition matrix" AND abs:graph AND abs:embedding',
        'ti:"k-step" AND abs:graph AND abs:representation',
    ],
    "pinsage": [
        'ti:"PinSage" OR ti:"PinSageConv"',
        'ti:"web-scale" AND ti:"recommendation" AND abs:graph',
        'ti:"random-walk" AND ti:"graph" AND abs:recommendation',
        'ti:"graph convolution" AND abs:recommendation AND abs:scalable',
    ],
    "fastgcn": [
        'ti:"FastGCN" OR ti:"fast graph convolution"',
        'ti:"sampling" AND ti:"graph convolution" AND abs:scalable',
        'ti:"fast" AND ti:"graph" AND abs:convolution AND abs:training',
        'ti:"stochastic" AND ti:"graph convolution"',
    ],
    "stochastic-gcn": [
        'ti:"stochastic training" AND ti:"graph convolution"',
        'ti:"variance reduction" AND abs:graph AND abs:convolution',
        'ti:"stochastic" AND ti:"graph" AND abs:training',
        'ti:"sampling" AND ti:"graph convolutional" AND abs:variance',
    ],
    "mhgcn": [
        'ti:"multiplex" AND ti:"heterogeneous" AND abs:graph',
        'ti:"multiplex" AND ti:"graph neural" AND abs:behavior',
        'ti:"heterogeneous" AND ti:"graph" AND abs:multiplex',
        'ti:"multiplex" AND ti:"graph" AND abs:representation',
    ],
    "uagsl": [
        'ti:"uncertainty-aware" AND ti:"graph structure"',
        'ti:"uncertainty" AND ti:"graph" AND abs:structure AND abs:learning',
        'ti:"graph structure learning" AND abs:uncertainty',
        'ti:"uncertainty" AND abs:graph AND abs:neural',
    ],
    "pi-gnn": [
        'ti:"physics-informed" AND ti:"graph neural"',
        'ti:"physics-informed" AND abs:graph AND abs:conservation',
        'ti:"physics" AND ti:"graph neural" AND abs:learning',
        'ti:"physics-informed" AND abs:graph AND abs:learning',
    ],
    "sdge": [
        'ti:"deep gaussian" AND ti:"graph" AND abs:embedding',
        'ti:"gaussian" AND ti:"graph" AND abs:embedding',
        'ti:"Gaussian" AND ti:"graph" AND abs:"distributional embedding"',
        'ti:"stochastic" AND ti:"graph" AND abs:embedding',
    ],
    "gde": [
        'ti:"generative distributional embedding"',
        'ti:"generative" AND ti:"distributional" AND abs:embedding',
        'ti:"graph" AND ti:"distributional" AND abs:embedding',
        'ti:"stochastic" AND ti:"graph" AND abs:embedding',
    ],
    "grrgnn": [
        'ti:"gated" AND ti:"residual" AND ti:"graph neural"',
        'ti:"recurrent" AND ti:"graph neural" AND abs:traffic',
        'ti:"gated" AND ti:"recurrent" AND abs:graph AND abs:neural',
        'ti:"residual" AND ti:"graph neural" AND abs:recurrent',
    ],
    "khg": [
        'ti:"knowledge hypergraph" OR ti:"hypergraph knowledge"',
        'ti:"hypergraph" AND abs:knowledge AND abs:prediction',
        'ti:"knowledge" AND ti:"hypergraph" AND abs:relation',
        'ti:"hypergraph" AND abs:"knowledge graph"',
    ],
    "shg": [
        'ti:"semantic hypergraph"',
        'ti:"semantic" AND ti:"hypergraph" AND abs:representation',
        'ti:"hypergraph" AND abs:semantic AND abs:learning',
        'ti:"semantic" AND ti:"graph" AND abs:hypergraph',
    ],
    "structuralseq-gnn": [
        'ti:"structural" AND ti:"sequence" AND abs:graph',
        'ti:"graph classification" AND ti:"structural attention"',
        'ti:"structural" AND ti:"attention" AND abs:graph AND abs:classification',
        'ti:"structure" AND ti:"sequence" AND abs:graph neural',
    ],
    "st-gdn": [
        'ti:"spatiotemporal" AND ti:"graph" AND abs:traffic',
        'ti:"spatio-temporal" AND ti:"graph" AND abs:prediction',
        'ti:"spatiotemporal" AND ti:"graph" AND abs:attention',
        'ti:"spatio-temporal" AND ti:"graph" AND abs:forecasting',
    ],
    "st-mgcn": [
        'ti:"spatiotemporal" AND ti:"multiscale" AND ti:"graph"',
        'ti:"spatio-temporal" AND ti:"multiscale" AND abs:graph',
        'ti:"multiscale" AND ti:"graph convolution" AND abs:spatiotemporal',
        'ti:"spatiotemporal" AND ti:"graph" AND abs:multiscale',
    ],
    "gmn": [
        'ti:"graph matching" AND ti:"neural" AND abs:similarity',
        'ti:"graph matching network"',
        'ti:"graph similarity" AND ti:"neural" AND abs:matching',
        'ti:"graph" AND ti:"matching" AND abs:"neural network"',
    ],
}


def query_arxiv(query, max_results=5):
    search_url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}"
    )
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            response = urllib.request.urlopen(req, timeout=45)
            xml_data = response.read().decode('utf-8')
            return parse_arxiv_xml(xml_data)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = BASE_DELAY * (2 ** attempt)
                print(f"    HTTP 429 - backing off {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP error {e.code}: {e}", flush=True)
                return []
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            time.sleep(BASE_DELAY)
            continue
    print(f"  Max retries reached: {query}", flush=True)
    return []


def parse_arxiv_xml(xml_data):
    papers = []
    try:
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
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
            title = re.sub(r'\s+', ' ', title_el.text.strip()) if title_el is not None and title_el.text else ''
            summary_el = entry.find('atom:summary', ns)
            summary = re.sub(r'\s+', ' ', summary_el.text.strip()) if summary_el is not None and summary_el.text else ''
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
                'arxiv_id': arxiv_id, 'title': title, 'summary': summary,
                'authors': authors, 'published': published, 'categories': categories,
                'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}", 'pdf_url': pdf_url,
            })
    except ET.ParseError as e:
        print(f"  XML parse error: {e}", flush=True)
    return papers


def get_existing_arxiv_ids(arch):
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
                content = f.read()
            ids.update(re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content))
            ids.update(re.findall(r'(\d{4}\.\d+)', fname))
        except:
            pass
    return ids


def sanitize_filename(title, arxiv_id):
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())[:80]
    return f"{clean}_{arxiv_id}.md"


def create_paper_md(paper, arch):
    authors_str = ", ".join(paper['authors'][:10])
    if len(paper['authors']) > 10:
        authors_str += " et al."
    year = paper['published'][:4] if paper['published'] else ""
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


def process_architecture(arch, queries, target_count=2):
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        print(f"  {arch}: papers dir missing, skipping", flush=True)
        return 0
    existing_ids = get_existing_arxiv_ids(arch)
    print(f"  {arch}: existing IDs = {existing_ids}", flush=True)
    collected = []
    seen_ids = set(existing_ids)
    for query in queries:
        if len(collected) >= target_count:
            break
        print(f"    Query: {query}", flush=True)
        results = query_arxiv(query, max_results=5)
        time.sleep(BASE_DELAY)
        for paper in results:
            if len(collected) >= target_count:
                break
            if paper['arxiv_id'] in seen_ids:
                continue
            if len(paper['title']) < 10:
                continue
            seen_ids.add(paper['arxiv_id'])
            collected.append(paper)
            print(f"    Found: {paper['arxiv_id']} - {paper['title'][:60]}", flush=True)
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


print("=== Starting graph arxiv fetch batch 2 ===", flush=True)
total_written = 0
for arch, queries in ARCH_QUERIES.items():
    print(f"\n=== Processing: {arch} ===", flush=True)
    papers_path = os.path.join(BASE, arch, "references", "papers")
    existing_count = len([f for f in os.listdir(papers_path) if f.endswith('.md')]) if os.path.isdir(papers_path) else 0
    needed = max(0, 4 - existing_count)
    needed = min(needed, 3)
    if needed == 0:
        print(f"  {arch}: already has {existing_count} papers, skipping", flush=True)
        continue
    written = process_architecture(arch, queries, target_count=needed)
    total_written += written
    print(f"  {arch}: {written} papers written", flush=True)

print(f"\n=== DONE: {total_written} total papers written ===", flush=True)
