#!/usr/bin/env python3
"""
Fetch related arXiv papers for vision architectures and convert to markdown.
Queries the arXiv API for related work beyond the primary paper already stored.
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

# arXiv namespace
NS = {"atom": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom"}

# Primary paper arXiv IDs (to exclude from results)
PRIMARY_IDS = {
    "3dgs": "2308.04079",
    "alexnet": None,
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
    "mask-r-cnn": None,
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

# Search queries for each architecture - using proper arXiv query syntax
# Format: ti:"phrase" for title, abs:"phrase" for abstract, AND/OR/ANDNOT
SEARCH_QUERIES = {
    # CNN families
    "resnet-50": [
        'ti:"residual" AND ti:"network" AND abs:"image classification"',
        'ti:"highway networks" AND abs:"deep"',
        'ti:"identity mappings" AND abs:"deep"',
    ],
    "alexnet": [
        'ti:"ImageNet" AND abs:"convolutional" AND abs:"classification"',
        'ti:"deep convolutional" AND abs:"ImageNet" AND abs:"classification"',
    ],
    "vgg-16": [
        'ti:"very deep convolutional" AND abs:"recognition"',
        'ti:"convolutional" AND abs:"ImageNet" AND abs:"depth" AND abs:"recognition"',
    ],
    "convnext-tiny": [
        'ti:"ConvNeXt" AND abs:"convolutional"',
        'ti:"modernized" AND abs:"convolutional" AND abs:"vision"',
        'ti:"convnet" AND abs:"transformer" AND abs:"design"',
    ],
    "mobilenet-v2": [
        'ti:"MobileNet" AND abs:"efficient"',
        'ti:"inverted residual" AND abs:"linear" AND abs:"bottleneck"',
        'ti:"depthwise separable" AND abs:"mobile"',
    ],
    "mobilenet-v3": [
        'ti:"MobileNetV3" AND abs:"efficient"',
        'ti:"mobile" AND abs:"hardware" AND abs:"efficient" AND abs:"architecture"',
        'ti:"squeeze and excitation" AND abs:"mobile"',
    ],
    "mobilenet-v4": [
        'ti:"MobileNetV4" AND abs:"universal"',
        'ti:"mobile" AND abs:"architecture" AND abs:"efficient"',
        'ti:"universal inverted bottleneck" AND abs:"mobile"',
    ],
    "efficientnet-b0": [
        'ti:"EfficientNet" AND abs:"compound"',
        'ti:"neural architecture search" AND abs:"scaling" AND abs:"convolutional"',
        'ti:"compound scaling" AND abs:"convolutional"',
    ],
    "densenet-121": [
        'ti:"DenseNet" AND abs:"dense connectivity"',
        'ti:"dense connectivity" AND abs:"convolutional"',
        'ti:"feature reuse" AND abs:"convolutional" AND abs:"network"',
    ],
    # ViT family
    "vit-b16": [
        'ti:"vision transformer" AND abs:"image" AND abs:"classification"',
        'ti:"ViT" AND abs:"transformer" AND abs:"vision"',
        'ti:"image transformer" AND abs:"attention" AND abs:"classification"',
    ],
    "deit": [
        'ti:"DeiT" AND abs:"transformer" AND abs:"vision"',
        'ti:"data-efficient" AND abs:"transformer" AND abs:"image"',
        'ti:"distillation" AND abs:"transformer" AND abs:"vision"',
    ],
    "dinov3": [
        'ti:"DINOv3" AND abs:"vision"',
        'ti:"self-supervised" AND abs:"vision" AND abs:"foundation" AND abs:"transformer"',
        'ti:"vision foundation model" AND abs:"self-supervised"',
    ],
    "din": [
        'ti:"deformable" AND ti:"image" AND abs:"recurrent"',
        'ti:"DIN" AND abs:"deformable" AND abs:"recurrent"',
        'ti:"deformable" AND abs:"recurrent" AND abs:"convolution"',
    ],
    "dino": [
        'ti:"DINO" AND abs:"self-supervised" AND abs:"vision"',
        'ti:"emerging properties" AND abs:"self-supervised" AND abs:"vision"',
        'ti:"self-supervised" AND abs:"vision transformer" AND abs:"emerging"',
    ],
    # Detection
    "yolo-v11": [
        'ti:"YOLO" AND abs:"real-time" AND abs:"detection"',
        'ti:"real-time object detection" AND abs:"single-stage"',
        'ti:"YOLOv" AND abs:"detection" AND abs:"real-time"',
    ],
    "yolo-v12": [
        'ti:"YOLO" AND abs:"real-time" AND abs:"detection" AND abs:"attention"',
        'ti:"YOLOv12" AND abs:"detection"',
        'ti:"attention-centric" AND abs:"detection" AND abs:"real-time"',
    ],
    "yolo26": [
        'ti:"YOLO" AND abs:"real-time" AND abs:"detection"',
        'ti:"YOLOv" AND abs:"object detection"',
        'ti:"real-time object detection" AND abs:"single-stage"',
    ],
    "detr": [
        'ti:"DETR" AND abs:"detection" AND abs:"transformer"',
        'ti:"deformable DETR" AND abs:"detection"',
        'ti:"end-to-end object detection" AND abs:"transformer"',
    ],
    "rtdetrv3": [
        'ti:"RT-DETR" AND abs:"real-time" AND abs:"detection"',
        'ti:"real-time detection transformer" AND abs:"DETR"',
        'ti:"RTDETR" AND abs:"detection"',
    ],
    "d-fine": [
        'ti:"D-FINE" AND abs:"detection"',
        'ti:"fine-grained" AND abs:"distribution" AND abs:"detection"',
        'ti:"object detection" AND abs:"distribution" AND abs:"refinement"',
    ],
    "rt-detr": [
        'ti:"RT-DETR" AND abs:"real-time" AND abs:"detection"',
        'ti:"real-time detection transformer" AND abs:"DETR"',
        'ti:"query selection" AND abs:"detection" AND abs:"transformer"',
    ],
    "grounding-dino": [
        'ti:"Grounding DINO" AND abs:"detection"',
        'ti:"open-set object detection" AND abs:"language"',
        'ti:"grounding" AND abs:"detection" AND abs:"text"',
    ],
    "mask-r-cnn": [
        'ti:"Mask R-CNN" AND abs:"instance" AND abs:"segmentation"',
        'ti:"instance segmentation" AND abs:"region" AND abs:"proposal"',
        'ti:"Faster R-CNN" AND abs:"instance" AND abs:"segmentation"',
    ],
    "mask2former": [
        'ti:"Mask2Former" AND abs:"segmentation"',
        'ti:"masked attention" AND abs:"segmentation" AND abs:"transformer"',
        'ti:"universal segmentation" AND abs:"transformer" AND abs:"mask"',
    ],
    "oneformer": [
        'ti:"OneFormer" AND abs:"segmentation"',
        'ti:"universal segmentation" AND abs:"single" AND abs:"transformer"',
        'ti:"panoptic" AND abs:"instance" AND abs:"semantic" AND abs:"segmentation"',
    ],
    # Segmentation
    "segformer": [
        'ti:"SegFormer" AND abs:"segmentation"',
        'ti:"semantic segmentation" AND abs:"transformer" AND abs:"hierarchical"',
        'ti:"segmentation transformer" AND abs:"efficient" AND abs:"encoder"',
    ],
    "sam": [
        'ti:"Segment Anything" AND abs:"segmentation" AND abs:"promptable"',
        'ti:"promptable segmentation" AND abs:"foundation"',
        'ti:"SAM" AND abs:"segmentation" AND abs:"foundation"',
    ],
    "sam2": [
        'ti:"SAM 2" AND abs:"segmentation" AND abs:"video"',
        'ti:"segment anything" AND abs:"video" AND abs:"promptable"',
        'ti:"video segmentation" AND abs:"promptable" AND abs:"foundation"',
    ],
    "sam3": [
        'ti:"SAM" AND abs:"segmentation" AND abs:"3D"',
        'ti:"segment anything" AND abs:"3D" AND abs:"scene"',
        'ti:"promptable segmentation" AND abs:"3D" AND abs:"video"',
    ],
    "sam-hq": [
        'ti:"SAM-HQ" AND abs:"segmentation" AND abs:"high-quality"',
        'ti:"high-quality segmentation" AND abs:"mask" AND abs:"anything"',
        'ti:"segment anything" AND abs:"quality" AND abs:"mask"',
    ],
    "fastsam": [
        'ti:"FastSAM" AND abs:"segmentation" AND abs:"fast"',
        'ti:"fast segment anything" AND abs:"YOLO"',
        'ti:"real-time segmentation" AND abs:"anything" AND abs:"fast"',
    ],
    "efficientsam": [
        'ti:"EfficientSAM" AND abs:"segmentation"',
        'ti:"efficient segment anything" AND abs:"mask"',
        'ti:"segmentation" AND abs:"efficient" AND abs:"promptable" AND abs:"SAM"',
    ],
    "mobilesam": [
        'ti:"MobileSAM" AND abs:"segmentation"',
        'ti:"mobile segment anything" AND abs:"efficient"',
        'ti:"lightweight segmentation" AND abs:"SAM" AND abs:"mobile"',
    ],
    "edgesam": [
        'ti:"EdgeSAM" AND abs:"segmentation" AND abs:"edge"',
        'ti:"edge device" AND abs:"segment anything" AND abs:"efficient"',
        'ti:"efficient segmentation" AND abs:"edge" AND abs:"device"',
    ],
    # Depth
    "depth-anything-v2": [
        'ti:"Depth Anything" AND abs:"monocular" AND abs:"depth"',
        'ti:"monocular depth" AND abs:"foundation" AND abs:"estimation"',
        'ti:"Depth Anything V2" AND abs:"depth"',
    ],
    "depth-pro": [
        'ti:"Depth Pro" AND abs:"monocular" AND abs:"depth"',
        'ti:"zero-shot metric depth" AND abs:"monocular"',
        'ti:"metric depth estimation" AND abs:"monocular"',
    ],
    "metric3d": [
        'ti:"Metric3D" AND abs:"monocular" AND abs:"depth"',
        'ti:"metric depth estimation" AND abs:"monocular" AND abs:"zero-shot"',
        'ti:"monocular depth" AND abs:"metric" AND abs:"canonical"',
    ],
    "unidepth": [
        'ti:"UniDepth" AND abs:"monocular" AND abs:"depth"',
        'ti:"universal monocular depth" AND abs:"estimation"',
        'ti:"monocular depth" AND abs:"3D" AND abs:"universal"',
    ],
    "videodepthanything": [
        'ti:"Video Depth Anything" AND abs:"video" AND abs:"depth"',
        'ti:"video depth estimation" AND abs:"monocular"',
        'ti:"video depth" AND abs:"temporal" AND abs:"consistent"',
    ],
    "depthcrafter": [
        'ti:"DepthCrafter" AND abs:"video" AND abs:"depth"',
        'ti:"video depth" AND abs:"generation" AND abs:"long"',
        'ti:"depth" AND abs:"video" AND abs:"diffusion" AND abs:"generation"',
    ],
    # Tracking
    "bytetrack": [
        'ti:"ByteTrack" AND abs:"multi-object" AND abs:"tracking"',
        'ti:"multi-object tracking" AND abs:"association" AND abs:"low-confidence"',
        'ti:"tracking" AND abs:"association" AND abs:"detection" AND abs:"multi-object"',
    ],
    "ocsort": [
        'ti:"OC-SORT" AND abs:"multi-object" AND abs:"tracking"',
        'ti:"observation-centric SORT" AND abs:"tracking"',
        'ti:"multi-object tracking" AND abs:"occlusion" AND abs:"SORT"',
    ],
    "tapir": [
        'ti:"TAPIR" AND abs:"point" AND abs:"tracking"',
        'ti:"tracking any point" AND abs:"video"',
        'ti:"point tracking" AND abs:"long-term" AND abs:"video"',
    ],
    "cotracker3": [
        'ti:"CoTracker" AND abs:"point" AND abs:"tracking"',
        'ti:"point tracking" AND abs:"video" AND abs:"long-term"',
        'ti:"tracking points" AND abs:"video" AND abs:"dense"',
    ],
    # 3D
    "3dgs": [
        'ti:"Gaussian Splatting" AND abs:"3D" AND abs:"rendering"',
        'ti:"3D Gaussian" AND abs:"radiance" AND abs:"field"',
        'ti:"Gaussian Splatting" AND abs:"real-time" AND abs:"rendering"',
    ],
    "nerf": [
        'ti:"NeRF" AND abs:"neural radiance field"',
        'ti:"neural radiance field" AND abs:"view synthesis"',
        'ti:"NeRF" AND abs:"novel view" AND abs:"synthesis"',
    ],
    "dust3r": [
        'ti:"DUSt3R" AND abs:"stereo" AND abs:"3D"',
        'ti:"dense unconstrained stereo" AND abs:"3D" AND abs:"reconstruction"',
        'ti:"stereo" AND abs:"3D" AND abs:"reconstruction" AND abs:"pointmaps"',
    ],
    "mast3r": [
        'ti:"MASt3R" AND abs:"stereo" AND abs:"3D"',
        'ti:"matching" AND abs:"stereo" AND abs:"3D" AND abs:"reconstruction"',
        'ti:"stereo matching" AND abs:"3D" AND abs:"reconstruction" AND abs:"pointmaps"',
    ],
    "vggt": [
        'ti:"VGGT" AND abs:"3D" AND abs:"reconstruction"',
        'ti:"geometry grounding" AND abs:"3D" AND abs:"reconstruction" AND abs:"transformer"',
        'ti:"feed-forward 3D" AND abs:"reconstruction" AND abs:"multi-view"',
    ],
    "monosplat": [
        'ti:"MonoSplat" AND abs:"3D" AND abs:"Gaussian"',
        'ti:"monocular Gaussian splatting" AND abs:"3D" AND abs:"reconstruction"',
        'ti:"3D Gaussian splatting" AND abs:"monocular" AND abs:"feed-forward"',
    ],
    "foundationstereo": [
        'ti:"FoundationStereo" AND abs:"stereo" AND abs:"matching"',
        'ti:"foundation stereo matching" AND abs:"3D" AND abs:"reconstruction"',
        'ti:"stereo matching" AND abs:"foundation" AND abs:"model" AND abs:"zero-shot"',
    ],
    "sea-raft": [
        'ti:"SEA-RAFT" AND abs:"optical" AND abs:"flow"',
        'ti:"optical flow" AND abs:"RAFT" AND abs:"estimation"',
        'ti:"RAFT" AND abs:"optical flow" AND abs:"improved"',
    ],
    "igev": [
        'ti:"IGEV" AND abs:"stereo" AND abs:"matching"',
        'ti:"geometry encoding volume" AND abs:"stereo" AND abs:"matching"',
        'ti:"stereo matching" AND abs:"cost volume" AND abs:"IGEV"',
    ],
    "gmflow": [
        'ti:"GMFlow" AND abs:"optical" AND abs:"flow"',
        'ti:"global matching" AND abs:"optical flow" AND abs:"transformer"',
        'ti:"optical flow" AND abs:"global" AND abs:"matching" AND abs:"transformer"',
    ],
    "raft": [
        'ti:"RAFT" AND abs:"optical" AND abs:"flow"',
        'ti:"recurrent all-pairs" AND abs:"optical flow"',
        'ti:"optical flow" AND abs:"recurrent" AND abs:"all-pairs"',
    ],
    "raft-stereo": [
        'ti:"RAFT-Stereo" AND abs:"stereo" AND abs:"matching"',
        'ti:"stereo RAFT" AND abs:"recurrent" AND abs:"matching"',
        'ti:"recurrent stereo matching" AND abs:"RAFT"',
    ],
}


def query_arxiv(search_query, max_results=8, retries=5):
    """Query arXiv API and return parsed entries."""
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AtlasResearchBot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
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
                
                results.append({
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
            return results
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited (429), waiting {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                print(f"  Attempt {attempt+1} failed: HTTP {e.code}: {e.reason}")
                time.sleep(10)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return []


def is_relevant(entry, arch_keywords, primary_id):
    """Check if a paper is relevant and not the primary paper."""
    if entry["arxiv_id"] == primary_id:
        return False
    if entry["arxiv_id"] in ALL_PRIMARY_IDS:
        return False
    
    vision_cats = {"cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.GR", "eess.IV"}
    if not any(c in vision_cats for c in entry["categories"]):
        return False
    
    text = (entry["title"] + " " + entry["summary"]).lower()
    for kw_group in arch_keywords:
        if all(kw.lower() in text for kw in kw_group):
            return True
    return False


RELEVANCE_KEYWORDS = {
    "resnet-50": [["residual", "network"], ["resnet"], ["deep", "network", "gradient"]],
    "alexnet": [["imagenet", "convolutional"], ["alexnet"], ["deep", "convolutional", "classification"]],
    "vgg-16": [["vgg"], ["very deep", "convolutional"], ["convolutional", "recognition", "depth"]],
    "convnext-tiny": [["convnext"], ["convnet", "transformer"], ["modernized", "convnet"]],
    "mobilenet-v2": [["mobilenet"], ["inverted", "residual"], ["depthwise", "separable"], ["mobile", "efficient"]],
    "mobilenet-v3": [["mobilenet"], ["mobile", "hardware"], ["squeeze", "excitation"]],
    "mobilenet-v4": [["mobilenet"], ["mobile", "universal"], ["universal inverted bottleneck"]],
    "efficientnet-b0": [["efficientnet"], ["compound", "scaling"], ["architecture", "search", "scaling"]],
    "densenet-121": [["densenet"], ["dense", "connectivity"], ["feature", "reuse"]],
    "vit-b16": [["vision", "transformer"], ["vit", "image"], ["transformer", "image", "classification"]],
    "deit": [["deit"], ["data-efficient", "transformer"], ["distillation", "transformer", "image"]],
    "dinov3": [["dinov3"], ["dino", "v3"], ["self-supervised", "foundation", "vision"], ["vision", "foundation", "model"]],
    "din": [["deformable", "image"], ["din", "recurrent"], ["deformable", "recurrent"]],
    "dino": [["dino", "self-supervised"], ["emerging", "properties", "self-supervised"], ["self-supervised", "vision", "transformer"]],
    "yolo-v11": [["yolo"], ["real-time", "detection"], ["single-stage", "detection"]],
    "yolo-v12": [["yolo"], ["real-time", "detection"], ["attention", "detection"]],
    "yolo26": [["yolo"], ["real-time", "detection"], ["object", "detection"]],
    "detr": [["detr"], ["detection", "transformer"], ["end-to-end", "detection"]],
    "rtdetrv3": [["rt-detr"], ["rtdetr"], ["real-time", "detection", "transformer"]],
    "d-fine": [["d-fine"], ["fine-grained", "detection"], ["distribution", "refinement", "detection"]],
    "rt-detr": [["rt-detr"], ["rtdetr"], ["real-time", "detection", "transformer"]],
    "grounding-dino": [["grounding", "dino"], ["open-set", "detection"], ["grounding", "detection", "text"]],
    "mask-r-cnn": [["mask", "r-cnn"], ["instance", "segmentation", "proposal"], ["faster", "r-cnn"]],
    "mask2former": [["mask2former"], ["masked", "attention", "segmentation"], ["universal", "segmentation", "transformer"]],
    "oneformer": [["oneformer"], ["universal", "segmentation"], ["panoptic", "instance", "semantic"]],
    "segformer": [["segformer"], ["semantic", "segmentation", "transformer"], ["segmentation", "transformer", "hierarchical"]],
    "sam": [["segment", "anything"], ["promptable", "segmentation"], ["sam", "segmentation", "foundation"]],
    "sam2": [["sam", "2"], ["segment", "anything", "video"], ["video", "segmentation", "promptable"]],
    "sam3": [["sam", "3d"], ["segment", "anything", "3d"], ["promptable", "3d", "segmentation"]],
    "sam-hq": [["sam-hq"], ["high-quality", "segmentation"], ["segment", "anything", "quality"]],
    "fastsam": [["fastsam"], ["fast", "segment", "anything"], ["real-time", "segmentation", "yolo"]],
    "efficientsam": [["efficientsam"], ["efficient", "segment", "anything"], ["segmentation", "efficient", "sam"]],
    "mobilesam": [["mobilesam"], ["mobile", "segment", "anything"], ["lightweight", "sam"]],
    "edgesam": [["edgesam"], ["edge", "segment", "anything"], ["efficient", "segmentation", "edge"]],
    "depth-anything-v2": [["depth", "anything"], ["monocular", "depth", "foundation"], ["depth", "anything", "v2"]],
    "depth-pro": [["depth", "pro"], ["zero-shot", "metric", "depth"], ["metric", "depth", "monocular"]],
    "metric3d": [["metric3d"], ["metric", "depth", "monocular"], ["monocular", "depth", "metric"]],
    "unidepth": [["unidepth"], ["universal", "monocular", "depth"], ["monocular", "depth", "3d"]],
    "videodepthanything": [["video", "depth", "anything"], ["video", "depth", "estimation"], ["video", "depth", "temporal"]],
    "depthcrafter": [["depthcrafter"], ["video", "depth", "generation"], ["depth", "video", "diffusion"]],
    "bytetrack": [["bytetrack"], ["multi-object", "tracking", "association"], ["tracking", "low-confidence"]],
    "ocsort": [["oc-sort"], ["observation-centric", "sort"], ["multi-object", "tracking", "occlusion"]],
    "tapir": [["tapir"], ["tracking", "any", "point"], ["point", "tracking", "video"]],
    "cotracker3": [["cotracker"], ["point", "tracking", "video"], ["tracking", "points", "dense"]],
    "3dgs": [["gaussian", "splatting"], ["3d", "gaussian"], ["gaussian", "radiance", "field"]],
    "nerf": [["nerf"], ["neural", "radiance", "field"], ["radiance", "field", "view"]],
    "dust3r": [["dust3r"], ["stereo", "3d", "reconstruction"], ["pointmaps", "dense"]],
    "mast3r": [["mast3r"], ["stereo", "matching", "3d"], ["pointmaps", "stereo"]],
    "vggt": [["vggt"], ["geometry", "grounding", "3d"], ["feed-forward", "3d", "reconstruction"]],
    "monosplat": [["monosplat"], ["monocular", "gaussian", "splatting"], ["3d", "gaussian", "monocular"]],
    "foundationstereo": [["foundationstereo"], ["foundation", "stereo", "matching"], ["stereo", "matching", "zero-shot"]],
    "sea-raft": [["sea-raft"], ["optical", "flow", "raft"], ["raft", "flow", "improved"]],
    "igev": [["igev"], ["geometry", "encoding", "volume", "stereo"], ["stereo", "matching", "cost", "volume"]],
    "gmflow": [["gmflow"], ["global", "matching", "flow"], ["optical", "flow", "global", "transformer"]],
    "raft": [["raft", "optical", "flow"], ["recurrent", "all-pairs", "field"], ["optical", "flow", "recurrent"]],
    "raft-stereo": [["raft-stereo"], ["stereo", "raft"], ["recurrent", "stereo", "matching"]],
}


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


def process_architecture(arch):
    """Process a single architecture: query arXiv, filter, generate markdown."""
    primary_id = PRIMARY_IDS.get(arch)
    queries = SEARCH_QUERIES.get(arch, [])
    keywords = RELEVANCE_KEYWORDS.get(arch, [])
    
    if not queries:
        print(f"  {arch}: No queries defined, skipping")
        return []
    
    all_results = []
    seen_ids = set()
    if primary_id:
        seen_ids.add(primary_id)
    seen_ids.update(ALL_PRIMARY_IDS)
    
    for query in queries:
        print(f"  Querying: {query[:80]}...")
        results = query_arxiv(query, max_results=10)
        time.sleep(5)
        
        for r in results:
            if r["arxiv_id"] not in seen_ids and is_relevant(r, keywords, primary_id):
                seen_ids.add(r["arxiv_id"])
                all_results.append(r)
                if len(all_results) >= 3:
                    break
        if len(all_results) >= 3:
            break
    
    papers_dir = os.path.join(ROOT, arch, "references", "papers")
    written = []
    for entry in all_results[:3]:
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
    
    archs = sorted(SEARCH_QUERIES.keys())
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
        time.sleep(5)
    
    log_path = "/Users/xiaming/Workspace/atlas/scripts/arxiv_fetch_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    total_written = 0
    for arch, papers in sorted(results_log.items()):
        count = len(papers)
        total_written += count
        status = "OK" if count >= 1 else "NEEDS ATTENTION"
        print(f"  {arch}: {count} papers written [{status}]")
    print(f"\nTotal papers written: {total_written}")


if __name__ == "__main__":
    main()
