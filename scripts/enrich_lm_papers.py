#!/usr/bin/env python3
"""
Enrich LM architectures with arXiv papers.
Queries arXiv API for related papers (excluding primary) and writes markdown
files in each architecture's references/papers/ directory.
Target 2-4 papers per architecture (aim for 3).
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import re
import json

ROOT = "/Users/xiaming/Workspace/atlas/architectures"
ARXIV_API = "https://export.arxiv.org/api/query"
TARGET_COUNT = 3  # target 2-4 papers, aim for 3

# Architecture -> (list of search queries, list of primary arxiv IDs to exclude)
ARCH_QUERIES = {
    # === LLM families with only primary paper ===
    "instructgpt": (
        ['ti:"InstructGPT" OR abs:"training language models to follow instructions"',
         'abs:"reinforcement learning from human feedback" language model align',
         'abs:"RLHF" alignment human feedback large language model'],
        []
    ),
    "mamba": (
        ['ti:"Mamba" abs:"state space model" language',
         'abs:"selective state space" sequence model language',
         'abs:"Mamba" language model transformer alternative'],
        ["2312.00752"]
    ),
    "mamba-3": (
        ['abs:"Mamba" language model improved',
         'abs:"state space model" language modeling',
         'ti:"Mamba" abs:"scaling" language'],
        []
    ),
    "localmamba": (
        ['ti:"LocalMamba" OR abs:"LocalMamba" visual recognition',
         'abs:"local attention" Mamba state space vision',
         'abs:"Mamba" vision hierarchical scanning'],
        ["2403.09338"]
    ),
    "vmamba": (
        ['ti:"VMamba" OR abs:"VMamba" visual state space',
         'abs:"visual state space model" Mamba',
         'abs:"cross-scan" Mamba vision transformer'],
        ["2401.10166"]
    ),
    "mambavision": (
        ['ti:"MambaVision" OR abs:"MambaVision"',
         'abs:"Mamba" vision image classification',
         'abs:"state space model" vision backbone image'],
        ["2407.08083"]
    ),
    "rwkv": (
        ['ti:"RWKV" OR abs:"RWKV" language model',
         'abs:"reinventing RNN" transformer era',
         'abs:"RWKV" linear attention recurrent'],
        ["2305.13048", "2305.13048"]
    ),
    "bert-base": (
        ['ti:"BERT" abs:"pre-training" bidirectional transformer',
         'abs:"masked language model" BERT pre-training',
         'abs:"bidirectional encoder" transformer representations BERT'],
        ["1810.04805"]
    ),
    "bert4rec": (
        ['ti:"BERT4Rec" OR abs:"BERT4Rec" sequential recommendation',
         'abs:"BERT" sequential recommendation session-based',
         'abs:"bidirectional" sequential recommendation transformer'],
        ["1904.06690"]
    ),
    "modernbert": (
        ['ti:"ModernBERT" OR abs:"ModernBERT"',
         'abs:"modern BERT" encoder transformer',
         'abs:"BERT" encoder improvements rotary embedding'],
        []
    ),
    "t5-small": (
        ['ti:"T5" abs:"text-to-text transfer transformer"',
         'abs:"text-to-text" transformer unified framework',
         'abs:"T5" exploring transfer learning language'],
        ["1910.10683"]
    ),
    "whisper-small": (
        ['ti:"Whisper" abs:"robust speech recognition"',
         'abs:"Whisper" weak supervision multilingual speech',
         'abs:"speech recognition" multilingual weakly supervised'],
        ["2212.04356"]
    ),
    "wav2vec2-base": (
        ['ti:"wav2vec 2.0" OR abs:"wav2vec 2.0" self-supervised speech',
         'abs:"self-supervised learning" speech representation wav2vec',
         'abs:"wav2vec" contrastive speech representation'],
        ["2006.11477"]
    ),
    "hubert-base": (
        ['ti:"HuBERT" OR abs:"HuBERT" self-supervised speech',
         'abs:"hidden unit BERT" speech representation',
         'abs:"self-supervised" speech representation clustering BERT'],
        ["2106.07447"]
    ),
    "differential-transformer": (
        ['ti:"Differential Transformer" OR abs:"differential transformer"',
         'abs:"differential attention" transformer language model',
         'abs:"differential" attention mechanism transformer'],
        []
    ),
    "megabyte": (
        ['ti:"MegaByte" OR abs:"MegaByte" transformer',
         'abs:"MegaByte" sub-quadratic transformer language model',
         'abs:"byte-level" transformer efficient long sequence'],
        ["2205.14141"]
    ),
    "xlstm": (
        ['ti:"xLSTM" OR abs:"xLSTM" extended long short-term memory',
         'abs:"extended LSTM" language model',
         'abs:"xLSTM" exponential gating memory'],
        ["2505.04517"]
    ),
    "s4": (
        ['ti:"S4" OR abs:"structured state space" sequence',
         'abs:"S4" efficient long sequence modeling',
         'abs:"structured state space" model HiPPO long-range'],
        ["2111.00358"]
    ),
    "transformer": (
        ['ti:"attention is all you need" OR abs:"attention is all you need"',
         'abs:"transformer" attention mechanism self-attention',
         'abs:"self-attention" transformer encoder decoder'],
        ["1706.03762"]
    ),
    "encodec": (
        ['ti:"EnCodec" OR abs:"EnCodec" neural audio codec',
         'abs:"neural audio codec" compression language model',
         'abs:"audio codec" residual vector quantization'],
        ["2210.13438"]
    ),
    "lwm": (
        ['ti:"Large World Model" OR abs:"Large World Model"',
         'abs:"large world model" video language understanding',
         'abs:"LWM" long context video language model'],
        ["2402.08268"]
    ),
    "samgpt": (
        ['abs:"SAM-GPT" OR abs:"SAM GPT" segmentation language',
         'abs:"segment anything" GPT graph foundation model',
         'abs:"text-free graph" foundation model multi-domain'],
        []
    ),
}


def query_arxiv(search_query, max_results=10, retries=2):
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
            print(f"    [retry {attempt+1}/{retries+1}] error: {last_err}", flush=True)
            if attempt < retries:
                time.sleep(5)
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


def slugify(s, maxlen=80):
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
    """Build markdown content for a paper entry, matching existing format."""
    title = entry['title']
    arxiv_url = entry['arxiv_url']
    authors = entry['authors']
    summary = entry['summary']
    published = entry['published']
    year = year_from_published(published)
    categories = entry['categories']
    comment = entry['comment']

    # Author short for header: "LastName et al. 2024"
    if authors:
        first_author_last = authors[0].split()[-1] if len(authors[0].split()) > 0 else authors[0]
        if len(authors) > 1:
            author_header = f"{first_author_last} et al. {year}"
        else:
            author_header = f"{first_author_last} {year}"
    else:
        author_header = f"Unknown {year}"

    lines = []
    lines.append(f"# Paper ({author_header})")
    lines.append("")
    lines.append(f"> Source: `{arxiv_url}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**{title}**")
    lines.append("")

    if authors:
        lines.append(", ".join(authors))
        lines.append("")

    lines.append("**Abstract**")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"- **arXiv ID**: `{entry['arxiv_id']}`")
    lines.append(f"- **Published**: {published[:10] if published else 'Unknown'}")
    if categories:
        lines.append(f"- **Categories**: {', '.join(categories)}")
    if entry['pdf_url']:
        lines.append(f"- **PDF**: {entry['pdf_url']}")
    lines.append("")

    return "\n".join(lines)


def get_existing_files(arch_dir):
    """Get set of existing markdown filenames and existing arxiv IDs in a papers dir."""
    existing_files = set()
    existing_ids = set()
    if os.path.isdir(arch_dir):
        for f in os.listdir(arch_dir):
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
        time.sleep(1)  # be polite to arXiv

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
            # Skip obviously irrelevant (physics papers, etc.)
            skip_titles = ['ILC Technology Network', 'Erratum', 'Comment on']
            if any(s in e['title'] for s in skip_titles):
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
            slug = slugify(e['title'], maxlen=60)
            filename = f"{author_s}_{year}_{e['arxiv_id']}.md"
            filepath = os.path.join(papers_dir, filename)

            # Avoid overwriting existing files
            if os.path.exists(filepath):
                print(f"  -> skip (exists): {filename}", flush=True)
                continue

            md = build_markdown(e)
            with open(filepath, 'w') as f:
                f.write(md)
            written += 1
            print(f"  -> wrote: {filename}", flush=True)

        results[arch] = written
        print(f"  => wrote {written} papers", flush=True)

    # Summary
    print("\n=== SUMMARY ===")
    total_written = sum(results.values())
    print(f"Total papers written: {total_written}")
    zero = [a for a, c in results.items() if c == 0]
    if zero:
        print(f"Architectures with 0 new papers: {zero}")

    # Save results for inspection
    with open("/tmp/enrich_lm_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
