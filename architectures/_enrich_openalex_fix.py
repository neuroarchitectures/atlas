#!/usr/bin/env python3
"""
Fix script: enrich architectures that got 0 papers in batch 2.
Uses broader/different search queries.
"""
import os, re, time, json, urllib.request, urllib.parse

BASE = "/Users/xiaming/Workspace/atlas/architectures"
OPENALEX_API = "https://api.openalex.org/works"
TARGET_TOTAL = 4

# Architectures that got 0 papers + missing ones, with broader queries
ARCH_QUERIES = {
    # Missing from batch 2 entirely
    "i-jepa": "image joint embedding predictive architecture self-supervised learning",
    "sne": "signed network embedding positive negative link prediction",
    "sne-enhanced": "enhanced network embedding structural equivalence graphlet",

    # Got 0 papers — query too specific
    "ctne": "context network embedding temporal dynamic graph representation",
    "tpdnr": "dynamic network representation learning temporal evolution",
    "tricl": "contrastive learning graph neural network self-supervised",
    "tnif": "neural inference framework transformer acceleration",
}


def query_openalex(search_query, max_results=15, retries=3):
    params = {
        'search': search_query,
        'per_page': str(max_results),
        'sort': 'relevance_score:desc',
    }
    url = OPENALEX_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'AtlasPaperEnricher/1.0 (mailto:research@atlas.local)',
        'Accept': 'application/json',
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            return parse_openalex_results(data)
        except Exception as e:
            print(f"    Error querying OpenAlex: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return []


def parse_openalex_results(data):
    papers = []
    for work in data.get('results', []):
        title = work.get('title', '') or work.get('display_name', '')
        if not title:
            continue
        title = re.sub(r'\s+', ' ', title.strip())

        doi = work.get('doi', '') or ''
        arxiv_id = ''
        locs = work.get('locations', [])
        for loc in locs:
            landing = loc.get('landing_page_url', '') or ''
            m = re.search(r'arxiv\.org/abs/(\d+\.\d+)', landing)
            if m:
                arxiv_id = m.group(1)
                break
            pdf = loc.get('pdf_url', '') or ''
            m = re.search(r'arxiv\.org/pdf/(\d+\.\d+)', pdf)
            if m:
                arxiv_id = m.group(1)
                break

        ids = work.get('ids', {})
        if not arxiv_id:
            for key in ('openalex', 'mag', 'doi'):
                val = ids.get(key, '') or ''
                m = re.search(r'(\d{4}\.\d{4,5})', val)
                if m:
                    arxiv_id = m.group(1)
                    break

        dedup_id = arxiv_id if arxiv_id else (doi if doi else ids.get('openalex', ''))

        abstract = ''
        inv_idx = work.get('abstract_inverted_index', {})
        if inv_idx:
            word_positions = []
            for word, positions in inv_idx.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = ' '.join(w for _, w in word_positions)

        authorships = work.get('authorships', [])
        authors = []
        for a in authorships:
            author = a.get('author', {})
            name = author.get('display_name', '')
            if name:
                authors.append(name)

        pub_year = work.get('publication_year', '')
        concepts = []
        for c in work.get('concepts', [])[:5]:
            name = c.get('display_name', '')
            if name:
                concepts.append(name)

        papers.append({
            'arxiv_id': arxiv_id,
            'doi': doi,
            'dedup_id': dedup_id,
            'title': title,
            'summary': abstract,
            'authors': authors,
            'pub_year': str(pub_year) if pub_year else '',
            'concepts': concepts,
        })
    return papers


def get_existing_ids(arch):
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
            matches = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content)
            ids.update(matches)
            fn_matches = re.findall(r'(\d{4}\.\d{4,5})', fname)
            ids.update(fn_matches)
            doi_matches = re.findall(r'doi\.org/(10\.\d+/[^\s`)]+)', content)
            ids.update(doi_matches)
        except:
            pass
    return ids


def get_existing_count(arch):
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        return 0
    return len([f for f in os.listdir(papers_path) if f.endswith('.md')])


def slugify(s, maxlen=80):
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:maxlen]


def build_markdown(paper):
    title = paper['title']
    arxiv_id = paper['arxiv_id']
    doi = paper['doi']
    authors = paper['authors']
    summary = paper['summary']
    pub_year = paper['pub_year']
    concepts = paper['concepts']

    if arxiv_id:
        source_url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        source_url = f"https://doi.org/{doi}"
    else:
        source_url = ""

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    if source_url:
        lines.append(f"> Source: `{source_url}`")
        lines.append("")
    lines.append("---")
    lines.append("")

    if authors:
        authors_str = ", ".join(authors[:10])
        if len(authors) > 10:
            authors_str += " et al."
        lines.append(f"**Authors:** {authors_str}")
        lines.append("")

    if pub_year and pub_year != 'None':
        lines.append(f"**Published:** {pub_year}")
        lines.append("")

    if concepts:
        lines.append(f"**Concepts:** {', '.join(concepts)}")
        lines.append("")

    if summary:
        lines.append("## Abstract")
        lines.append("")
        lines.append(summary)
        lines.append("")

    if arxiv_id:
        lines.append(f"**PDF:** https://arxiv.org/pdf/{arxiv_id}")
    elif doi:
        lines.append(f"**DOI:** https://doi.org/{doi}")
        lines.append("")

    return "\n".join(lines)


def process_architecture(arch, query, target_total=TARGET_TOTAL):
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        os.makedirs(papers_path, exist_ok=True)

    existing_ids = get_existing_ids(arch)
    existing_count = get_existing_count(arch)
    need = max(0, target_total - existing_count)

    print(f"\n=== Processing: {arch} ===")
    print(f"  {arch}: existing={existing_count}, need={need} more, existing IDs={existing_ids}")

    if need == 0:
        print(f"  {arch}: already has enough papers, skipping")
        return 0

    print(f"    Query: {query}")
    results = query_openalex(query, max_results=15)
    time.sleep(1)

    collected = []
    seen = set(existing_ids)

    for paper in results:
        if len(collected) >= need:
            break
        dedup = paper['dedup_id']
        if not dedup:
            dedup = paper['title'][:50]
        if dedup in seen:
            continue
        if len(paper['title']) < 10:
            continue
        seen.add(dedup)
        collected.append(paper)
        if paper['arxiv_id']:
            print(f"    Found: {paper['arxiv_id']} - {paper['title'][:60]}")
        elif paper['doi']:
            print(f"    Found: {paper['doi']} - {paper['title'][:60]}")
        else:
            print(f"    Found: {paper['title'][:60]}")

    written = 0
    for paper in collected:
        slug = slugify(paper['title'], maxlen=60)
        if paper['arxiv_id']:
            filename = f"{slug}_{paper['arxiv_id']}.md"
        elif paper['doi']:
            filename = f"{slug}.md"
        else:
            filename = f"{slug}.md"
        fpath = os.path.join(papers_path, filename)
        if os.path.exists(fpath):
            filename = f"Related_{slug}.md"
            fpath = os.path.join(papers_path, filename)
        with open(fpath, 'w') as f:
            f.write(build_markdown(paper))
        written += 1
        print(f"    Written: {filename}")

    print(f"  {arch}: {written} papers written")
    return written


def main():
    total_written = 0
    for arch, query in ARCH_QUERIES.items():
        written = process_architecture(arch, query)
        total_written += written

    print(f"\n=== DONE: {total_written} total papers written ===")


if __name__ == '__main__':
    main()
