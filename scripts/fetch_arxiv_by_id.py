#!/usr/bin/env python3
"""
Fetch related arXiv papers for vision architectures using ID-based API queries.
Uses curated lists of known related paper arXiv IDs per architecture.
The ID-based API (id_list=...) is much less rate-limited than search queries.
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

# Primary paper arXiv IDs (to exclude from results)
PRIMARY_IDS = {
    "3dgs": "2308.04079",
    "alexnet": None,  # 1404.5997 not in filename
    "bytetrack": "2110.06864",
    "convnext-tiny": "2201.03545",
    "cotracker3": "2410.11831",
    "d-fine": "2410.13842",
    "deit": "2012.12877",
    "densenet-121": "1608.06993",
    "depth-anything-v2": "2406.09414",
    "depth-pro": "2410.02073",
    "depthcrafter": "2409.02095",
    "detr": "2005.12872",
    "din": "1706.06978",
    "dino": "2203.03605",
    "dinov3": "2508.10104",
    "dust3r": "2312.14132",
    "edgesam": "2312.06660",
    "efficientnet-b0": "1905.11946",
    "efficientsam": "2312.00863",
    "fastsam": "2306.12156",
    "foundationstereo": "2501.09898",
    "gmflow": "2111.13680",
    "grounding-dino": "2303.05499",
    "igev": "2303.06615",
    "mask-r-cnn": None,  # 1703.06870 not in filename
    "mask2former": "2112.01527",
    "mast3r": "2406.09756",
    "metric3d": "2307.10984",
    "mobilenet-v2": "1801.04381",
    "mobilenet-v3": "1905.02244",
    "mobilenet-v4": "2404.10518",
    "mobilesam": "2306.14289",
    "monosplat": "2505.15185",
    "nerf": "2003.08934",
    "ocsort": "2203.14360",
    "oneformer": "2211.06220",
    "raft": "2003.12039",
    "raft-stereo": "2109.07547",
    "resnet-50": "1512.03385",
    "rt-detr": "2304.08069",
    "rtdetrv3": "2409.08475",
    "sam": "2304.02643",
    "sam-hq": "2306.01567",
    "sam2": "2408.00714",
    "sam3": "2511.16719",
    "sea-raft": "2405.14793",
    "segformer": "2105.15203",
    "tapir": "2306.08637",
    "unidepth": "2403.18913",
    "vgg-16": "1409.1556",
    "vggt": "2503.11651",
    "videodepthanything": "2501.12375",
    "vit-b16": "2010.11929",
    "yolo-v11": "2410.17725",
    "yolo-v12": "2502.12524",
    "yolo26": "2606.03748",
}

ALL_PRIMARY_IDS = set(v for v in PRIMARY_IDS.values() if v)

# Curated related arXiv paper IDs for each architecture.
# These are well-known related/follow-up papers in the same research area.
# We provide more than needed (5-8) so after filtering, 2-3 remain.
RELATED_IDS = {
    # CNN families
    "resnet-50": [
        "1605.07146",   # Identity Mappings in Deep Residual Networks (ResNet v2)
        "1606.01505",   # Highway Networks
        "1511.06422",   # Aggregated Residual Transformations (ResNeXt)
        "1709.01507",   # Squeeze-and-Excitation Networks (SENet)
        "1812.01187",   # Bag of Tricks for Image Classification (ResNet tricks)
    ],
    "alexnet": [
        "1409.1556",    # VGG (very deep convolutional networks) - related
        "1603.05279",   # XNOR-Net
        "1506.02509",   # BinaryConnect
        "1512.03385",   # ResNet (successor)
    ],
    "vgg-16": [
        "1409.4842",    # GoogLeNet/Inception
        "1512.03385",   # ResNet
        "1605.07146",   # ResNet v2
        "1602.07360",   # Inception-v4
    ],
    "convnext-tiny": [
        "2103.14030",   # Swin Transformer
        "2204.02977",   # CoAtNet
        "2303.16900",   # ConvNeXt-V2
        "2201.03545",   # (primary, will be filtered)
    ],
    "mobilenet-v2": [
        "1704.04861",   # MobileNetV1
        "1905.02244",   # MobileNetV3
        "1801.04381",   # (primary)
        "1807.03784",   # MnasNet
        "1905.11946",   # EfficientNet
    ],
    "mobilenet-v3": [
        "1801.04381",   # MobileNetV2
        "2404.10518",   # MobileNetV4
        "1807.03784",   # MnasNet
        "1905.11946",   # EfficientNet
    ],
    "mobilenet-v4": [
        "1905.02244",   # MobileNetV3
        "1801.04381",   # MobileNetV2
        "2204.06406",   # MobileViT
        "2106.10270",   # EfficientNetV2
    ],
    "efficientnet-b0": [
        "1905.11946",   # (primary)
        "2106.10270",   # EfficientNetV2
        "1807.03784",   # MnasNet
        "1801.04381",   # MobileNetV2
        "1704.04861",   # MobileNetV1
    ],
    "densenet-121": [
        "1512.03385",   # ResNet
        "1605.07146",   # ResNet v2
        "1608.06993",   # (primary)
        "1707.01629",   # Dual Path Networks (DPN)
        "1709.01507",   # SENet
    ],
    # ViT family
    "vit-b16": [
        "2010.11929",   # (primary)
        "2103.14030",   # Swin Transformer
        "2012.12877",   # DeiT
        "2106.04540",   # CSWin Transformer
        "2203.03605",   # DINO
    ],
    "deit": [
        "2010.11929",   # ViT
        "2103.14030",   # Swin
        "2012.12877",   # (primary)
        "2204.02977",   # CoAtNet
    ],
    "dinov3": [
        "2104.14294",   # DINO (original)
        "2304.07193",   # DINOv2
        "2508.10104",   # (primary)
        "2301.11222",   # MAE
    ],
    "din": [
        "1703.06211",   # Deformable Convolutional Networks (DCN)
        "1706.06978",   # (primary)
        "1811.11468",   # Deformable ConvNets v2
        "1612.00542",   # Deformable Part-based Model
    ],
    "dino": [
        "2010.11929",   # ViT
        "2006.07733",   # Bootstrap Your Own Latent (BYOL)
        "2203.03605",   # (primary)
        "2301.11222",   # MAE
        "1911.05722",   # SimCLR
    ],
    # Detection
    "yolo-v11": [
        "2410.17725",   # (primary)
        "2502.12524",   # YOLOv12
        "2402.13616",   # YOLOv9
        "2307.11538",   # YOLOv8 (Ultralytics)
    ],
    "yolo-v12": [
        "2502.12524",   # (primary)
        "2410.17725",   # YOLOv11
        "2402.13616",   # YOLOv9
        "2307.11538",   # YOLOv8
    ],
    "yolo26": [
        "2606.03748",   # (primary)
        "2502.12524",   # YOLOv12
        "2410.17725",   # YOLOv11
        "2402.13616",   # YOLOv9
    ],
    "detr": [
        "2005.12872",   # (primary)
        "2010.04159",   # Deformable DETR
        "2103.01800",   # Conditional DETR
        "2101.06378",   # TSP-FCOS / UP-DETR
        "2203.03605",   # DINO (detection)
    ],
    "rtdetrv3": [
        "2409.08475",   # (primary)
        "2304.08069",   # RT-DETR
        "2410.13842",   # D-FINE
        "2307.07923",   # DINO-DETR detection
    ],
    "d-fine": [
        "2410.13842",   # (primary)
        "2409.08475",   # RT-DETRv3
        "2304.08069",   # RT-DETR
        "2203.03605",   # DINO
    ],
    "rt-detr": [
        "2304.08069",   # (primary)
        "2409.08475",   # RT-DETRv3
        "2005.12872",   # DETR
        "2410.13842",   # D-FINE
        "2203.03605",   # DINO
    ],
    "grounding-dino": [
        "2303.05499",   # (primary)
        "2305.02411",   # Grounding DINO 1.5 / Pro
        "2112.05912",   # GLIP
        "2304.08069",   # RT-DETR
    ],
    "mask-r-cnn": [
        "1506.01497",   # Faster R-CNN
        "1703.06870",   # Mask R-CNN (primary, ID not in filename)
        "1612.03144",   # FPN
        "1901.07518",   # FCOS
        "2005.12872",   # DETR
    ],
    "mask2former": [
        "2112.01527",   # (primary)
        "2012.00747",   # MaskFormer
        "2203.03605",   # DINO
        "2005.12872",   # DETR
    ],
    "oneformer": [
        "2211.06220",   # (primary)
        "2112.01527",   # Mask2Former
        "2012.00747",   # MaskFormer
        "1502.02766",   # Panoptic Segmentation
    ],
    # Segmentation
    "segformer": [
        "2105.15203",   # (primary)
        "1909.11916",   # HRNet
        "2007.04269",   # SETR (Segmentation Transformer)
        "2103.14030",   # Swin
    ],
    "sam": [
        "2304.02643",   # (primary)
        "2306.14289",   # MobileSAM
        "2306.12156",   # FastSAM
        "2306.01567",   # SAM-HQ
        "2305.02411",   # Grounding DINO
    ],
    "sam2": [
        "2408.00714",   # (primary)
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2306.12156",   # FastSAM
    ],
    "sam3": [
        "2511.16719",   # (primary)
        "2408.00714",   # SAM2
        "2304.02643",   # SAM
        "2306.01567",   # SAM-HQ
    ],
    "sam-hq": [
        "2306.01567",   # (primary)
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2312.06660",   # EdgeSAM
    ],
    "fastsam": [
        "2306.12156",   # (primary)
        "2304.02643",   # SAM
        "2306.14289",   # MobileSAM
        "2312.00863",   # EfficientSAM
    ],
    "efficientsam": [
        "2312.00863",   # (primary)
        "2304.02643",   # SAM
        "2306.12156",   # FastSAM
        "2306.14289",   # MobileSAM
    ],
    "mobilesam": [
        "2306.14289",   # (primary)
        "2304.02643",   # SAM
        "2306.12156",   # FastSAM
        "2312.06660",   # EdgeSAM
    ],
    "edgesam": [
        "2312.06660",   # (primary)
        "2306.14289",   # MobileSAM
        "2304.02643",   # SAM
        "2312.00863",   # EfficientSAM
    ],
    # Depth
    "depth-anything-v2": [
        "2406.09414",   # (primary)
        "2312.02123",   # Depth Anything V1
        "2307.10984",   # Metric3D
        "2403.18913",   # UniDepth
    ],
    "depth-pro": [
        "2410.02073",   # (primary)
        "2307.10984",   # Metric3D
        "2406.09414",   # Depth Anything V2
        "2403.18913",   # UniDepth
    ],
    "metric3d": [
        "2307.10984",   # (primary)
        "2403.18913",   # UniDepth
        "2406.09414",   # Depth Anything V2
        "2410.02073",   # Depth Pro
    ],
    "unidepth": [
        "2403.18913",   # (primary)
        "2307.10984",   # Metric3D
        "2410.02073",   # Depth Pro
        "2406.09414",   # Depth Anything V2
    ],
    "videodepthanything": [
        "2501.12375",   # (primary)
        "2406.09414",   # Depth Anything V2
        "2409.02095",   # DepthCrafter
        "2312.02123",   # Depth Anything V1
    ],
    "depthcrafter": [
        "2409.02095",   # (primary)
        "2501.12375",   # Video Depth Anything
        "2406.09414",   # Depth Anything V2
        "2312.02123",   # Depth Anything V1
    ],
    # Tracking
    "bytetrack": [
        "2110.06864",   # (primary)
        "2203.14360",   # OC-SORT
        "2306.08637",   # TAPIR
        "2410.11831",   # CoTracker3
    ],
    "ocsort": [
        "2203.14360",   # (primary)
        "2110.06864",   # ByteTrack
        "2306.08637",   # TAPIR
        "2410.11831",   # CoTracker3
    ],
    "tapir": [
        "2306.08637",   # (primary)
        "2410.11831",   # CoTracker3
        "2110.06864",   # ByteTrack
        "2307.07635",   # CoTracker (v1)
    ],
    "cotracker3": [
        "2410.11831",   # (primary)
        "2306.08637",   # TAPIR
        "2307.07635",   # CoTracker v1
        "2110.06864",   # ByteTrack
    ],
    # 3D
    "3dgs": [
        "2308.04079",   # (primary)
        "2003.08934",   # NeRF
        "2406.02720",   # 4D Gaussian Splatting
        "2401.00834",   # Deblurring 3DGS
        "2403.11134",   # Gaussian Opacity Fields
    ],
    "nerf": [
        "2003.08934",   # (primary)
        "2308.04079",   # 3DGS
        "2103.10325",   # Mip-NeRF
        "2201.12500",   # Instant-NGP
    ],
    "dust3r": [
        "2312.14132",   # (primary)
        "2406.09756",   # MASt3R
        "2503.11651",   # VGGT
        "2308.04079",   # 3DGS
    ],
    "mast3r": [
        "2406.09756",   # (primary)
        "2312.14132",   # DUSt3R
        "2503.11651",   # VGGT
        "2505.15185",   # MonoSplat
    ],
    "vggt": [
        "2503.11651",   # (primary)
        "2312.14132",   # DUSt3R
        "2406.09756",   # MASt3R
        "2505.15185",   # MonoSplat
    ],
    "monosplat": [
        "2505.15185",   # (primary)
        "2503.11651",   # VGGT
        "2308.04079",   # 3DGS
        "2406.09756",   # MASt3R
    ],
    "foundationstereo": [
        "2501.09898",   # (primary)
        "2312.14132",   # DUSt3R
        "2406.09756",   # MASt3R
        "2503.11651",   # VGGT
    ],
    "sea-raft": [
        "2405.14793",   # (primary)
        "2003.12039",   # RAFT
        "2111.13680",   # GMFlow
        "2303.06615",   # IGEV-Stereo
    ],
    "igev": [
        "2303.06615",   # (primary)
        "2109.07547",   # RAFT-Stereo
        "2003.12039",   # RAFT
        "2405.14793",   # SEA-RAFT
    ],
    "gmflow": [
        "2111.13680",   # (primary)
        "2003.12039",   # RAFT
        "2405.14793",   # SEA-RAFT
        "2204.02670",   # FlowFormer
    ],
    "raft": [
        "2003.12039",   # (primary)
        "2111.13680",   # GMFlow
        "2405.14793",   # SEA-RAFT
        "2109.07547",   # RAFT-Stereo
    ],
    "raft-stereo": [
        "2109.07547",   # (primary)
        "2003.12039",   # RAFT
        "2303.06615",   # IGEV-Stereo
        "2405.14793",   # SEA-RAFT
    ],
}


def fetch_papers_by_id(id_list, retries=5):
    """Fetch paper metadata from arXiv using id_list query (less rate-limited)."""
    if not id_list:
        return []
    
    # arXiv API allows up to ~20 IDs per request
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
                break  # Success, no retry needed
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
        
        time.sleep(4)  # Rate limit between batches
    
    return all_results


def generate_filename(entry, arch):
    """Generate a markdown filename following the existing convention."""
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
    """Generate markdown content for a paper."""
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
    """Get set of arXiv IDs already present in the papers directory."""
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
    """Process a single architecture: fetch related papers, filter, generate markdown."""
    primary_id = PRIMARY_IDS.get(arch)
    related_ids = RELATED_IDS.get(arch, [])
    
    if not related_ids:
        print(f"  {arch}: No related IDs defined, skipping")
        return []
    
    # Get existing paper IDs to avoid duplicates
    existing_ids = get_existing_paper_ids(arch)
    
    # Filter out primary and already-existing papers
    ids_to_fetch = []
    for rid in related_ids:
        if rid == primary_id:
            continue
        if rid in existing_ids:
            continue
        if rid in ALL_PRIMARY_IDS:
            continue
        ids_to_fetch.append(rid)
    
    if not ids_to_fetch:
        print(f"  {arch}: All related papers already exist or are primary, skipping")
        return []
    
    print(f"  Fetching {len(ids_to_fetch)} papers: {ids_to_fetch}")
    results = fetch_papers_by_id(ids_to_fetch)
    
    # Filter results: only keep papers that were actually found
    found_ids = {r["arxiv_id"] for r in results}
    
    # Sort results by the order in ids_to_fetch
    id_order = {aid: i for i, aid in enumerate(ids_to_fetch)}
    results.sort(key=lambda r: id_order.get(r["arxiv_id"], 999))
    
    # Take up to 3 papers (to bring total to 2-4 with the primary)
    papers_dir = os.path.join(ROOT, arch, "references", "papers")
    existing_count = len([f for f in os.listdir(papers_dir) if f.endswith('.md')]) if os.path.isdir(papers_dir) else 0
    max_new = max(0, 4 - existing_count)
    
    written = []
    for entry in results[:max_new]:
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
    
    archs = sorted(RELATED_IDS.keys())
    print(f"Processing {len(archs)} architectures...\n")
    
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
        time.sleep(3)
    
    log_path = "/Users/xiaming/Workspace/atlas/scripts/arxiv_fetch_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    total_written = 0
    for arch, papers in sorted(results_log.items()):
        count = len(papers)
        total_written += count
        # Final count check
        pd = os.path.join(ROOT, arch, "references", "papers")
        final_count = len([f for f in os.listdir(pd) if f.endswith('.md')]) if os.path.isdir(pd) else 0
        status = "OK" if final_count >= 2 else "NEEDS ATTENTION"
        print(f"  {arch}: {count} new papers written, {final_count} total [{status}]")
    print(f"\nTotal new papers written: {total_written}")


if __name__ == "__main__":
    main()
