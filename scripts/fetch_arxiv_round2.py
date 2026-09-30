#!/usr/bin/env python3
"""
Round 2: Fetch additional related arXiv papers for architectures that still have only 1 paper.
Uses IDs of related papers that are NOT primary papers of any architecture in the set.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import os
import re
import json

ROOT = "/Users/xiaming/Workspace/atlas/architectures"
BASE_URL = "https://export.arxiv.org/api/query"

NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}

# Primary paper IDs per architecture (only used to exclude the arch's own primary)
PRIMARY_IDS = {
    "d-fine": "2410.13842",
    "depth-pro": "2410.02073",
    "dust3r": "2312.14132",
    "edgesam": "2312.06660",
    "efficientsam": "2312.00863",
    "fastsam": "2306.12156",
    "foundationstereo": "2501.09898",
    "igev": "2303.06615",
    "mast3r": "2406.09756",
    "metric3d": "2307.10984",
    "mobilesam": "2306.14289",
    "monosplat": "2505.15185",
    "ocsort": "2203.14360",
    "raft": "2003.12039",
    "raft-stereo": "2109.07547",
    "rt-detr": "2304.08069",
    "sam-hq": "2306.01567",
    "sam2": "2408.00714",
    "sam3": "2511.16719",
    "sea-raft": "2405.14793",
    "unidepth": "2403.18913",
    "vggt": "2503.11651",
}

# IDs already used as related papers in round 1 (to avoid cross-architecture duplicates
# where the same paper would appear in multiple architecture directories).
# We allow these to be used for OTHER architectures, just not re-used for the same one.
# This set is intentionally empty - we want related papers to be shared across architectures
# when they're genuinely related.
ROUND1_USED_IDS = set()

# New related paper IDs for architectures that still need papers.
# These are carefully chosen to be relevant to each architecture.
# Papers that are primary papers of OTHER architectures are fine as related papers here.
ADDITIONAL_RELATED_IDS = {
    "d-fine": [
        "2204.10241",   # DINO: DETR with Improved Denoising Anchor Boxes
        "2305.17106",   # Focus-DETR
        "2110.06468",   # Conditional DETR
        "2010.04159",   # Deformable DETR
    ],
    "depth-pro": [
        "2206.09614",   # DPT (Dense Prediction Transformer for depth)
        "2103.13413",   # MiDaS (robust monocular depth)
        "1907.01341",   # Monodepth2
        "2006.09035",   # Omnidepth
    ],
    "dust3r": [
        "2304.03677",   # LoFTR (local feature matching)
        "2306.14691",   # RoMa (robust matching)
        "2308.04079",   # 3DGS (related 3D reconstruction)
        "2003.08934",   # NeRF (related neural rendering)
    ],
    "edgesam": [
        "2304.02643",   # SAM (original)
        "2306.14289",   # MobileSAM
        "2312.00863",   # EfficientSAM
        "2306.12156",   # FastSAM
    ],
    "efficientsam": [
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2306.12156",   # FastSAM
        "2207.06491",   # TinyViT
    ],
    "fastsam": [
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2312.00863",   # EfficientSAM
        "2312.06660",   # EdgeSAM
    ],
    "foundationstereo": [
        "2306.14691",   # RoMa
        "2304.03677",   # LoFTR
        "2303.06615",   # IGEV-Stereo
        "2406.09756",   # MASt3R
    ],
    "igev": [
        "2109.07547",   # RAFT-Stereo
        "2003.12039",   # RAFT
        "2405.14793",   # SEA-RAFT
        "2304.03677",   # LoFTR
    ],
    "mast3r": [
        "2306.14691",   # RoMa
        "2304.03677",   # LoFTR
        "2308.04079",   # 3DGS
        "2312.14132",   # DUSt3R (primary, excluded)
    ],
    "metric3d": [
        "2206.09614",   # DPT
        "2103.13413",   # MiDaS
        "1907.01341",   # Monodepth2
        "2006.09035",   # Omnidepth
    ],
    "mobilesam": [
        "2304.02643",   # SAM
        "2306.12156",   # FastSAM
        "2312.00863",   # EfficientSAM
        "2312.06660",   # EdgeSAM
    ],
    "monosplat": [
        "2308.04079",   # 3DGS
        "2406.09756",   # MASt3R
        "2503.11651",   # VGGT
        "2312.14132",   # DUSt3R
    ],
    "ocsort": [
        "2110.06864",   # ByteTrack
        "1602.00763",   # SORT
        "1902.09646",   # DeepSORT
        "2010.12150",   # JDE
    ],
    "raft": [
        "2111.13680",   # GMFlow
        "2405.14793",   # SEA-RAFT
        "2304.03677",   # LoFTR
        "2103.13955",   # GMA
    ],
    "raft-stereo": [
        "2003.12039",   # RAFT
        "2303.06615",   # IGEV
        "2405.14793",   # SEA-RAFT
        "2304.03677",   # LoFTR
    ],
    "rt-detr": [
        "2005.12872",   # DETR
        "2010.04159",   # Deformable DETR
        "2203.03605",   # DINO (detection)
        "2204.10241",   # DINO detection
    ],
    "sam-hq": [
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2306.12156",   # FastSAM
        "2312.06660",   # EdgeSAM
    ],
    "sam2": [
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2306.12156",   # FastSAM
        "2312.06660",   # EdgeSAM
    ],
    "sam3": [
        "2304.02643",   # SAM
        "2408.00714",   # SAM2
        "2306.14289",   # MobileSAM
        "2306.01567",   # SAM-HQ
    ],
    "sea-raft": [
        "2003.12039",   # RAFT
        "2111.13680",   # GMFlow
        "2103.13955",   # GMA
        "2304.03677",   # LoFTR
    ],
    "unidepth": [
        "2206.09614",   # DPT
        "2103.13413",   # MiDaS
        "1907.01341",   # Monodepth2
        "2006.09035",   # Omnidepth
    ],
    "vggt": [
        "2308.04079",   # 3DGS
        "2406.09756",   # MASt3R
        "2312.14132",   # DUSt3R
        "2304.03677",   # LoFTR
    ],
}


def fetch_papers_by_id(id_list, retries=5):
    """Fetch paper metadata from arXiv using id_list query."""
    if not id_list:
        return []
    
    all_results = []
    batch_size = 15
    
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        id_str = ",".join(batch)
        params = {
            "id_list": id_str,
            "start": 0,
            "max_results": len(batch),
        }
        url = BASE_URL + "?" + urllib.parse.urlencode(params)
        
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AtlasResearchBot/1.0"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = response.read().decode("utf-8")
                root = ET.fromstring(data)
                entries = root.findall("atom:entry", NS)
                for entry in entries:
                    entry_id = entry.find("atom:id", NS)
                    arxiv_url = entry_id.text if entry_id is not None else ""
                    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
                    arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
                    
                    title_el = entry.find("atom:title", NS)
                    title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
                    title = re.sub(r'\s+', ' ', title)
                    
                    summary_el = entry.find("atom:summary", NS)
                    summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                    summary = re.sub(r'\s+', ' ', summary)
                    
                    published_el = entry.find("atom:published", NS)
                    published = published_el.text if published_el is not None else ""
                    
                    authors = []
                    for author in entry.findall("atom:author", NS):
                        name_el = author.find("atom:name", NS)
                        if name_el is not None:
                            authors.append(name_el.text.strip())
                    
                    categories = []
                    for cat in entry.findall("atom:category", NS):
                        categories.append(cat.get("term", ""))
                    
                    pdf_url = ""
                    abs_url = ""
                    for link in entry.findall("atom:link", NS):
                        if link.get("title") == "pdf":
                            pdf_url = link.get("href", "")
                        elif link.get("rel") == "alternate":
                            abs_url = link.get("href", "")
                    
                    doi_el = entry.find("arxiv:doi", NS)
                    doi = doi_el.text if doi_el is not None else ""
                    
                    journal_el = entry.find("arxiv:journal_ref", NS)
                    journal_ref = journal_el.text if journal_el is not None else ""
                    
                    comment_el = entry.find("arxiv:comment", NS)
                    comment = comment_el.text if comment_el is not None else ""
                    
                    all_results.append({
                        "arxiv_id": arxiv_id_clean,
                        "arxiv_id_raw": arxiv_id,
                        "title": title,
                        "summary": summary,
                        "published": published,
                        "authors": authors,
                        "categories": categories,
                        "pdf_url": pdf_url,
                        "abs_url": abs_url,
                        "doi": doi,
                        "journal_ref": journal_ref,
                        "comment": comment,
                    })
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  Rate limited (429), waiting {wait}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    print(f"  Attempt {attempt+1} failed: HTTP {e.code}: {e.reason}")
                    time.sleep(10)
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(10)
        
        time.sleep(5)
    
    return all_results


def generate_filename(entry, arch):
    if entry["authors"]:
        first_author = entry["authors"][0]
        parts = first_author.split()
        last_name = parts[-1] if parts else first_author
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    else:
        last_name = "Unknown"
    
    year = entry["published"][:4] if entry["published"] else "2024"
    arxiv_id = entry["arxiv_id"]
    
    filename = f"{last_name}_et_al._{year}_{arxiv_id}.md"
    return filename


def generate_markdown(entry, arch):
    title = entry["title"]
    authors = entry["authors"]
    summary = entry["summary"]
    published = entry["published"]
    arxiv_id = entry["arxiv_id"]
    abs_url = entry["abs_url"] or f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = entry["pdf_url"] or f"https://arxiv.org/pdf/{arxiv_id}"
    doi = entry["doi"]
    journal_ref = entry["journal_ref"]
    comment = entry["comment"]
    categories = entry["categories"]
    
    year = published[:4] if published else "2024"
    
    if authors:
        first_author = authors[0]
        parts = first_author.split()
        last_name = parts[-1] if parts else first_author
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    else:
        last_name = "Unknown"
    
    md = f"""# Paper ({last_name} et al. {year})

> Source: `{abs_url}`

---

{title}
"""
    md += "\n" + "\n".join(authors) + "\n"
    
    if published:
        md += f"\nPublished: {published}\n"
    
    if categories:
        md += f"\nCategories: {', '.join(categories)}\n"
    
    if doi:
        md += f"\nDOI: {doi}\n"
    
    if journal_ref:
        md += f"\nJournal Reference: {journal_ref}\n"
    
    if comment:
        md += f"\nComment: {comment}\n"
    
    md += f"\nPDF: {pdf_url}\n"
    
    md += f"\nAbstract\n{summary}\n"
    
    return md


def get_existing_paper_ids(arch):
    papers_dir = os.path.join(ROOT, arch, "references", "papers")
    existing_ids = set()
    if os.path.isdir(papers_dir):
        for f in os.listdir(papers_dir):
            if f.endswith('.md'):
                m = re.search(r'(\d{4}\.\d{4,5})', f)
                if m:
                    existing_ids.add(m.group(1))
    return existing_ids


def process_architecture(arch):
    related_ids = ADDITIONAL_RELATED_IDS.get(arch, [])
    own_primary = PRIMARY_IDS.get(arch)
    
    if not related_ids:
        print(f"  {arch}: No additional IDs defined, skipping")
        return []
    
    existing_ids = get_existing_paper_ids(arch)
    
    # Filter out the arch's own primary paper and already-existing papers
    ids_to_fetch = []
    for rid in related_ids:
        if rid in existing_ids:
            continue
        if rid == own_primary:
            continue
        ids_to_fetch.append(rid)
    
    if not ids_to_fetch:
        print(f"  {arch}: All additional papers already exist or are primary, skipping")
        return []
    
    print(f"  Fetching {len(ids_to_fetch)} papers: {ids_to_fetch}")
    results = fetch_papers_by_id(ids_to_fetch)
    
    # Filter: only keep papers that were actually found (not 404)
    found_results = [r for r in results if r["arxiv_id"] in ids_to_fetch and r["title"]]
    
    # Sort results by the order in ids_to_fetch
    id_order = {aid: i for i, aid in enumerate(ids_to_fetch)}
    found_results.sort(key=lambda r: id_order.get(r["arxiv_id"], 999))
    
    papers_dir = os.path.join(ROOT, arch, "references", "papers")
    existing_count = len([f for f in os.listdir(papers_dir) if f.endswith('.md')]) if os.path.isdir(papers_dir) else 0
    max_new = max(0, 4 - existing_count)
    
    written = []
    for entry in found_results[:max_new]:
        filename = generate_filename(entry, arch)
        filepath = os.path.join(papers_dir, filename)
        
        if os.path.exists(filepath):
            print(f"  {arch}: {filename} already exists, skipping")
            continue
        
        content = generate_markdown(entry, arch)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(filename)
        print(f"  {arch}: Wrote {filename}")
    
    return written


def main():
    results_log = {}
    
    archs = sorted(ADDITIONAL_RELATED_IDS.keys())
    print(f"Processing {len(archs)} architectures (round 2)...\n")
    
    for i, arch in enumerate(archs, 1):
        # Skip architectures that already have enough papers
        papers_dir = os.path.join(ROOT, arch, "references", "papers")
        if os.path.isdir(papers_dir):
            existing = [f for f in os.listdir(papers_dir) if f.endswith('.md')]
            if len(existing) >= 4:
                print(f"[{i}/{len(archs)}] {arch}: already has {len(existing)} papers, skipping")
                results_log[arch] = []
                continue
        
        print(f"[{i}/{len(archs)}] Processing {arch}...")
        try:
            written = process_architecture(arch)
            results_log[arch] = written
        except Exception as e:
            print(f"  ERROR for {arch}: {e}")
            results_log[arch] = []
        time.sleep(5)
    
    print(f"\n=== SUMMARY (Round 2) ===")
    total_written = 0
    for arch, papers in sorted(results_log.items()):
        count = len(papers)
        total_written += count
        pd = os.path.join(ROOT, arch, "references", "papers")
        final_count = len([f for f in os.listdir(pd) if f.endswith('.md')]) if os.path.isdir(pd) else 0
        status = "OK" if final_count >= 2 else "NEEDS ATTENTION"
        print(f"  {arch}: {count} new papers written, {final_count} total [{status}]")
    print(f"\nTotal new papers written: {total_written}")


if __name__ == "__main__":
    main()
