#!/usr/bin/env python3
"""
Fetch related arXiv papers for each architecture and write markdown files.
Queries arXiv API, selects 2-4 related papers per architecture (excluding primary),
and writes markdown into each references/papers/ directory.
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
    # ===== GRAPH =====
    "gcn": (['ti:"graph convolutional"', 'abs:"graph convolutional network" semi-supervised'], ["1609.02907"]),
    "graphsage": (['ti:"GraphSAGE"', 'abs:"inductive representation learning" large graphs'], ["1706.02216"]),
    "gat": (['ti:"graph attention"', 'abs:"graph attention network"'], ["1710.10903"]),
    "graph-transformer": (['ti:"graph transformer"', 'abs:"graph transformer" attention'], ["2012.09699"]),
    "lightgcn": (['ti:"LightGCN"', 'abs:"light graph convolutional" collaborative filtering'], ["2002.02126"]),
    "netgan": (['ti:"NetGAN"', 'abs:"NetGAN" graph generation'], ["1802.03687"]),
    "graphvae": (['ti:"GraphVAE"', 'abs:"variational autoencoder" graph generation'], ["1611.07308"]),
    "molgan": (['ti:"MolGAN"', 'abs:"molecular graph" generative adversarial'], ["1805.11973"]),
    "dgi": (['ti:"Deep Graph Infomax"', 'abs:"deep graph infomax" unsupervised'], ["1809.10341"]),
    "cane": (['ti:"context-aware network embedding"', 'abs:"CANE" relation modeling'], ["1603.01718"]),
    "cognn": (['abs:"cooperative graph neural"', 'abs:"cooperative graph neural network"'], ["2308.03632"]),
    "hgemb": (['abs:"heterogeneous graph embedding"', 'ti:"heterogeneous graph embedding"'], ["2305.01844"]),
    "metapath2vec": (['ti:"metapath2vec"', 'abs:"metapath2vec" heterogeneous network'], ["1707.00394"]),
    "struc2vec": (['ti:"struc2vec"', 'abs:"struc2vec" structural identity'], ["1704.03124"]),
    "hin2vec": (['ti:"HIN2Vec"', 'abs:"HIN2Vec" heterogeneous information network'], ["1707.03522"]),
    "grarep": (['ti:"GraRep"', 'abs:"GraRep" graph representation learning'], ["1506.02236"]),
    "pinsage": (['ti:"PinSage"', 'abs:"PinSage" graph convolutional recommendation'], ["1806.01973"]),
    "fastgcn": (['ti:"FastGCN"', 'abs:"fast graph convolutional" sampling'], ["1801.10247"]),
    "stochastic-gcn": (['ti:"stochastic training" graph convolutional', 'abs:"GraphSAGE" stochastic training'], ["1710.10568"]),
    "mhgcn": (['abs:"multi-relational graph convolutional"', 'abs:"multi-relational" graph neural network embedding'], ["2305.01844"]),
    "uagsl": (['abs:"uncertainty-aware" graph structure learning', 'abs:"uncertainty aware" graph learning'], ["2305.01844"]),
    "pi-gnn": (['abs:"position-invariant graph neural"', 'ti:"position-invariant" graph neural'], ["2305.01844"]),
    "sdge": (['abs:"scalable dynamic graph embedding"', 'abs:"scalable" dynamic graph representation'], ["2305.01844"]),
    "gde": (['ti:"graph dynamical embedding"', 'abs:"continuous-time dynamic graph" neural network'], ["2305.01844"]),
    "grrgnn": (['abs:"graph regularized" recurrent neural', 'abs:"graph recurrent" neural network'], ["2305.01844"]),
    "khg": (['abs:"knowledge-enhanced heterogeneous graph"', 'abs:"knowledge" heterogeneous graph neural'], ["2305.01844"]),
    "shg": (['abs:"simplified heterogeneous graph"', 'abs:"simplified" heterogeneous graph neural network'], ["2305.01844"]),
    "structuralseq-gnn": (['abs:"structural sequence" graph neural', 'abs:"structural" sequence graph network'], ["2305.01844"]),
    "st-gdn": (['ti:"graph deviation network"', 'abs:"spatiotemporal" graph deviation network'], ["2305.01844"]),
    "st-mgcn": (['abs:"spatiotemporal multi-graph convolutional"', 'abs:"spatiotemporal" graph convolutional network traffic'], ["2305.01844"]),
    "gmn": (['ti:"Graph Matching Network"', 'abs:"graph matching network"'], ["1903.03543"]),

    # ===== RECSYS =====
    "ncf": (['ti:"neural collaborative filtering"', 'abs:"neural collaborative filtering"'], ["1708.05031"]),
    "neumf": (['ti:"Neural Matrix Factorization"', 'abs:"neural matrix factorization" recommendation'], ["1708.05031"]),
    "deepfm": (['ti:"DeepFM"', 'abs:"factorization machine" neural network CTR'], ["1703.04247"]),
    "wide-and-deep": (['ti:"Wide & Deep"', 'abs:"wide and deep learning" recommender'], ["1606.07792"]),
    "din": (['ti:"Deep Interest Network"', 'abs:"deep interest network" CTR'], ["1706.06978"]),
    "dien": (['ti:"Deep Interest Evolution Network"', 'abs:"deep interest evolution network"'], ["1809.03672"]),
    "bst": (['ti:"Behavior Sequence Transformer"', 'abs:"behavior sequence transformer" recommendation'], ["1905.06874"]),
    "gru4rec": (['ti:"GRU4Rec"', 'abs:"session-based recommendations" recurrent neural'], ["1511.06939"]),
    "sasrec": (['ti:"Self-Attentive Sequential Recommendation"', 'abs:"self-attentive" sequential recommendation'], ["1808.09781"]),
    "mmoe": (['ti:"Multi-Gate Mixture-of-Experts"', 'abs:"mixture of experts" multi-task learning'], ["1705.02343"]),
    "ple": (['ti:"Progressive Layered Extraction"', 'abs:"progressive layered extraction" multi-task'], ["2005.08100"]),
    "esmm": (['ti:"Entire Space Multi-Task Model"', 'abs:"entire space" multi-task model'], ["1804.07931"]),
    "two-tower": (['abs:"two-tower" neural network recommendation', 'abs:"dual tower" retrieval model'], ["1906.03309"]),
    "dlrm": (['ti:"Deep Learning Recommendation Model"', 'abs:"deep learning recommendation model" DLRM'], ["1906.00091"]),
    "masa": (['abs:"memory augmented" sequential recommendation', 'abs:"MASA" recommendation'], ["2406.04221"]),
    "harp": (['abs:"hierarchical" representation recommendation', 'abs:"HARP" recommendation'], ["1606.07792"]),
    "msde": (['abs:"multi-scale" dynamic embedding recommendation', 'abs:"multi-scale dynamic" embedding'], ["2305.01844"]),

    # ===== MULTIMODAL =====
    "clip-vit-b32": (['ti:"CLIP" OR ti:"contrastive language-image"', 'abs:"contrastive language-image" pretraining'], ["2103.00020"]),
    "siglip-base": (['ti:"SigLip"', 'abs:"sigmoid loss" language image pretraining'], ["2303.15343"]),
    "blip2": (['ti:"BLIP-2"', 'abs:"BLIP-2" bootstrapping language image'], ["2301.12597"]),
    "llava-1.5-7b": (['ti:"LLaVA"', 'abs:"visual instruction tuning" large language'], ["2310.03744"]),
    "flamingo": (['ti:"Flamingo"', 'abs:"Flamingo" visual language model'], ["2204.14198"]),
    "uni-perceiver": (['ti:"Uni-Perceiver"', 'abs:"unified perception" multimodal pretraining'], ["2112.01522"]),
    "perception-encoder": (['abs:"perception encoder" multimodal', 'abs:"perception-encoder" vision'], ["2504.13181"]),
    "videoprism": (['ti:"VideoPrism"', 'abs:"video foundation model" encoder'], ["2402.13217"]),

    # ===== VIDEO =====
    "videomae": (['ti:"VideoMAE"', 'abs:"masked autoencoder" video self-supervised'], ["2203.12602"]),
    "timesformer": (['ti:"TimeSformer"', 'abs:"divided space-time attention" video'], ["2102.05095"]),
    "vivit": (['ti:"ViViT"', 'abs:"video vision transformer"'], ["2103.15691"]),
    "slowfast": (['ti:"SlowFast"', 'abs:"SlowFast networks" video recognition'], ["1812.03982"]),
    "v-jepa": (['ti:"V-JEPA"', 'abs:"video joint embedding predictive" architecture'], ["2404.08471"]),
    "v-jepa-2": (['ti:"V-JEPA 2"', 'abs:"video joint embedding predictive" v-jepa'], ["2506.09985"]),
    "v-jepa-2-ac": (['abs:"video joint embedding predictive" action conditioned', 'abs:"V-JEPA" action'], ["2506.09985"]),
    "i-jepa": (['ti:"I-JEPA"', 'abs:"image joint embedding predictive" architecture'], ["2301.08243"]),
    "depthcrafter": (['ti:"DepthCrafter"', 'abs:"DepthCrafter" video depth'], ["2409.02095"]),

    # ===== POSE =====
    "vitpose": (['ti:"ViTPose"', 'abs:"vision transformer" pose estimation'], ["2204.12484"]),
    "rtmpose": (['ti:"RTMPose"', 'abs:"real-time" pose estimation mmpose'], ["2303.07399"]),
    "hrnet": (['ti:"HRNet"', 'abs:"deep high-resolution representation" human pose'], ["1908.07919"]),

    # ===== MEDICAL =====
    "eegnet": (['ti:"EEGNet"', 'abs:"EEGNet" electroencephalogram convolutional'], ["1611.08024"]),
    "eeg-conformer": (['abs:"EEG Conformer" convolutional attention', 'abs:"EEG" conformer" classification'], ["2305.01844"]),
    "patch-tst": (['ti:"PatchTST"', 'abs:"patch time series transformer"'], ["2211.14730"]),
    "timesnet": (['ti:"TimesNet"', 'abs:"TimesNet" temporal 2D variation'], ["2210.02186"]),

    # ===== OTHER =====
    "dreamer-v3": (['ti:"DreamerV3"', 'abs:"DreamerV3" world model reinforcement'], ["2301.04104"]),
    "branchynet": (['ti:"BranchyNet"', 'abs:"early exiting" deep neural network'], ["1709.01686"]),
    "once-for-all": (['ti:"Once-for-All"', 'abs:"once for all" neural network training'], ["1908.09791"]),
    "slimmable": (['ti:"slimmable neural"', 'abs:"slimmable" neural network'], ["1812.08928"]),
    "ghostnet": (['ti:"GhostNet"', 'abs:"GhostNet" cheap operations'], ["1911.11907"]),
    "shufflenet-v2": (['ti:"ShuffleNet V2"', 'abs:"ShuffleNet" efficient architecture'], ["1807.11164"]),
    "maxvit": (['ti:"MaxViT"', 'abs:"MaxViT" multi-axis vision transformer'], ["2204.01697"]),
    "mobilevit": (['ti:"MobileViT"', 'abs:"MobileViT" mobile vision transformer'], ["2110.02178"]),
    "efficientvit": (['ti:"EfficientViT"', 'abs:"EfficientViT" efficient vision transformer'], ["2205.14756"]),
    "repvit": (['ti:"RepViT"', 'abs:"RepViT" reparameterized vision transformer'], ["2307.09283"]),
    "starnet": (['ti:"StarNet"', 'abs:"StarNet" efficient neural network'], ["2403.19967"]),
    "fasternet": (['ti:"FasterNet"', 'abs:"FasterNet" partial convolution'], ["2303.03667"]),
    "pvt": (['ti:"Pyramid Vision Transformer"', 'abs:"pyramid vision transformer"'], ["2102.12122"]),
    "swin-tiny": (['ti:"Swin Transformer"', 'abs:"Swin Transformer" hierarchical vision'], ["2103.14030"]),
    "swin-v2": (['ti:"Swin Transformer V2"', 'abs:"Swin Transformer V2" scaled up'], ["2111.09883"]),
    "cswin": (['ti:"CSWin Transformer"', 'abs:"CSWin" cross-shaped window transformer'], ["2107.00652"]),
    "deit": (['ti:"DeiT"', 'abs:"data-efficient image transformers" distillation'], ["2012.12877"]),
    "crate": (['abs:"white box transformer" sparse rate reduction', 'abs:"CRATE" transformer'], ["2306.01197"]),
    "fff": (['abs:"fast feedforward" neural network tree', 'abs:"FFF" fast feedforward'], ["2305.01844"]),
    "lcm": (['ti:"Latent Consistency Models"', 'abs:"latent consistency model" diffusion'], ["2310.04332"]),
    "block-diffusion": (['ti:"Block Diffusion"', 'abs:"block diffusion" language model'], ["2503.09573"]),
    "diffusion-unet": (['abs:"denoising diffusion probabilistic" U-Net', 'abs:"diffusion model" U-Net architecture'], ["2112.10752"]),
    "dit-xl2": (['ti:"Diffusion Transformers"', 'abs:"DiT" diffusion transformer"'], ["2212.09748"]),
    "unet": (['ti:"U-Net"', 'abs:"U-Net" convolutional biomedical segmentation'], ["1505.04597"]),
    "nerf": (['ti:"NeRF"', 'abs:"neural radiance field" view synthesis'], ["2003.08934"]),
    "3dgs": (['ti:"3D Gaussian Splatting"', 'abs:"3D Gaussian" radiance field rendering'], ["2308.04079"]),
    "geocalib": (['ti:"GeoCalib"', 'abs:"geometric calibration" camera pose'], ["2409.06704"]),
    "samgpt": (['abs:"SAM-GPT" segment anything', 'abs:"SAM GPT" segmentation language'], ["2305.01844"]),
}


def query_arxiv(search_query, max_results=8, retries=2):
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
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = resp.read().decode('utf-8')
            return parse_arxiv_response(data)
        except Exception as e:
            last_err = str(e)
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
    # clean
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
            # Skip obviously irrelevant (e.g., physics ILC report)
            if 'ILC Technology Network' in e['title']:
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
                continue

            md = build_markdown(e)
            with open(filepath, 'w') as f:
                f.write(md)
            written += 1

        results[arch] = written
        print(f"  -> wrote {written} papers", flush=True)

    # Summary
    print("\n=== SUMMARY ===")
    total_written = sum(results.values())
    print(f"Total papers written: {total_written}")
    zero = [a for a, c in results.items() if c == 0]
    if zero:
        print(f"Architectures with 0 new papers: {zero}")

    # Save results for inspection
    with open("/tmp/fetch_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
