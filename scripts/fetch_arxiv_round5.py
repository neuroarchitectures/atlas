#!/usr/bin/env python3
"""Round 5: Fetch papers for 9 architectures that still need them."""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import os
import time

ROOT = "/Users/xiaming/Workspace/atlas/architectures"
BASE_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Verified correct arXiv IDs for related papers
RELATED_IDS = {
    "depth-anything-v2": ["2312.02145", "2307.10984", "2103.13413"],
    "depthcrafter": ["2312.02145", "2406.09414", "2103.13413"],
    "grounding-dino": ["2304.02643", "2005.12872", "2104.14294"],
    "mask2former": ["2012.00747", "2005.12872", "2103.14030"],
    "mobilenet-v3": ["1704.04861", "1801.04381", "1905.11946"],
    "nerf": ["2308.04079", "2312.14132", "2003.12039"],
    "rtdetrv3": ["2304.08069", "2005.12872", "2010.04159"],
    "sam": ["2306.14289", "2306.12156", "2312.00863"],
    "videodepthanything": ["2312.02145", "2406.09414", "2409.02095"],
}

def get_existing_ids(arch):
    papers_dir = os.path.join(ROOT, arch, "references", "papers")
    ids = set()
    if os.path.isdir(papers_dir):
        for f in os.listdir(papers_dir):
            m = re.search(r'(\d{4}\.\d{4,5})', f)
            if m:
                ids.add(m.group(1))
    return ids

def fetch_papers_by_id(ids):
    ids_str = ",".join(ids)
    url = f"{BASE_URL}?id_list={ids_str}&max_results={len(ids)}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AtlasResearchBot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            root = ET.fromstring(data)
            entries = root.findall("atom:entry", NS)
            results = []
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
                results.append({"arxiv_id": arxiv_id_clean, "title": title, "summary": summary,
                    "published": published, "authors": authors, "categories": categories,
                    "pdf_url": pdf_url, "abs_url": abs_url, "doi": doi,
                    "journal_ref": journal_ref, "comment": comment})
            return results
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            if "429" in str(e):
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                time.sleep(5)
    return []

def generate_filename(entry):
    if entry["authors"]:
        first_author = entry["authors"][0]
        parts = first_author.split()
        last_name = parts[-1] if parts else first_author
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    else:
        last_name = "Unknown"
    year = entry["published"][:4] if entry["published"] else "2024"
    return f"{last_name}_et_al._{year}_{entry['arxiv_id']}.md"

def generate_markdown(entry):
    title = entry["title"]
    authors = entry["authors"]
    summary = entry["summary"]
    published = entry["published"]
    arxiv_id = entry["arxiv_id"]
    abs_url = entry["abs_url"] or f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = entry["pdf_url"] or f"https://arxiv.org/pdf/{arxiv_id}"
    year = published[:4] if published else "2024"
    if authors:
        first_author = authors[0]
        parts = first_author.split()
        last_name = parts[-1] if parts else first_author
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
    else:
        last_name = "Unknown"
    md = f"# Paper ({last_name} et al. {year})\n\n> Source: `{abs_url}`\n\n---\n\n{title}\n"
    md += "\n" + "\n".join(authors) + "\n"
    if published:
        md += f"\nPublished: {published}\n"
    if entry["categories"]:
        md += f"\nCategories: {', '.join(entry['categories'])}\n"
    if entry["doi"]:
        md += f"\nDOI: {entry['doi']}\n"
    if entry["journal_ref"]:
        md += f"\nJournal Reference: {entry['journal_ref']}\n"
    if entry["comment"]:
        md += f"\nComment: {entry['comment']}\n"
    md += f"\nPDF: {pdf_url}\n"
    md += f"\nAbstract\n{summary}\n"
    return md

def main():
    print("Round 5: Fetching papers for 9 architectures...", flush=True)
    for arch, ids in RELATED_IDS.items():
        existing = get_existing_ids(arch)
        to_fetch = [i for i in ids if i not in existing]
        if not to_fetch:
            print(f"  {arch}: all already exist", flush=True)
            continue
        print(f"  {arch}: fetching {to_fetch}", flush=True)
        results = fetch_papers_by_id(to_fetch)
        time.sleep(4)
        papers_dir = os.path.join(ROOT, arch, "references", "papers")
        for entry in results:
            if entry["arxiv_id"] in to_fetch:
                fname = generate_filename(entry)
                fpath = os.path.join(papers_dir, fname)
                if not os.path.exists(fpath):
                    content = generate_markdown(entry)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"    Wrote: {fname} - {entry['title'][:60]}", flush=True)
                else:
                    print(f"    Exists: {fname}", flush=True)
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
