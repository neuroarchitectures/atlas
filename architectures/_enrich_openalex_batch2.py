#!/usr/bin/env python3
"""
Enrich architectures with papers via OpenAlex API (batch 2).
Queries OpenAlex for related papers and writes markdown files.
OpenAlex has no aggressive rate limiting (unlike arXiv).
Target: bring each architecture to 3+ papers in references/papers/.
"""
import os
import re
import time
import json
import urllib.request
import urllib.parse

BASE = "/Users/xiaming/Workspace/atlas/architectures"
OPENALEX_API = "https://api.openalex.org/works"
TARGET_TOTAL = 4  # aim for 4 total (including existing)

# Architecture -> search query string for OpenAlex
# Built from model name + key domain terms from architecture.md descriptions
ARCH_QUERIES = {
    # === NLP / LLM ===
    "all-minilm-l6": "MiniLM sentence embedding distillation BERT",
    "armt": "ARMT attentive recurrent memory transformer long sequence",
    "bge-base-en": "BGE general embedding text retrieval BAAI",
    "cldlm": "continuous diffusion language model text generation",
    "cnn-lstm-1d": "CNN LSTM hybrid physiological signal ECG classification",
    "fast-transformer-decoder": "fast transformer decoder linear attention recurrent",
    "lstm": "long short-term memory recurrent neural network Hochreiter",
    "lwm": "large world model long context video language",
    "nsa": "native sparse attention deepseek efficient long context",
    "rmt": "recurrent memory transformer long context",
    "samgpt": "SAM-GPT segmentation anything model language",
    "simple-rnn": "recurrent neural network vanishing gradient time series",

    # === Computer Vision ===
    "bevdepth": "BEVDepth camera bird eye view 3D detection depth",
    "bevformer": "BEVFormer bird eye view transformer 3D perception autonomous driving",
    "branchynet": "BranchyNet early exit deep neural network inference",
    "centerpoint": "CenterPoint 3D object detection point cloud autonomous driving",
    "crate": "CRATE white box transformer sparse rate reduction interpretable",
    "cswin": "CSWin Transformer cross-shaped window attention vision",
    "deformable-detr": "Deformable DETR deformable attention object detection",
    "eeg-conformer": "EEG Conformer convolution attention brain computer interface",
    "eegnet": "EEGNet compact convolutional neural network EEG brain computer interface",
    "efficientvit": "EfficientViT efficient vision transformer segmentation",
    "fasternet": "FasterNet partial convolution efficient network design",
    "fff": "fast feedforward network tree neural network efficient",
    "flowformer": "FlowFormer optical flow transformer attention",
    "geocalib": "GeoCalib camera calibration single image vanishing point",
    "ghostnet": "GhostNet cheap operations efficient CNN mobile",
    "hrnet": "HRNet high resolution network pose estimation segmentation",
    "maxvit": "MaxViT multi-axis vision transformer attention",
    "mobilevit": "MobileViT mobile vision transformer lightweight",
    "moge": "MoGe monocular geometry estimation 3D scene",
    "mvit-v2": "MViTv2 multiscale vision transformer video recognition",
    "once-for-all": "Once-for-All network efficient adaptive deployment",
    "perception-encoder": "perception encoder multimodal vision foundation",
    "petr": "PETR position embedding transformation 3D detection",
    "ple": "progressive layered extraction multi-task learning recommendation",
    "pvt": "Pyramid Vision Transformer hierarchical vision backbone",
    "repvit": "RepViT reparameterized vision transformer efficient mobile",
    "rpn": "region proposal network object detection Faster R-CNN",
    "rpn-v2": "region proposal network improved object detection",
    "rtmpose": "RTMPose real-time pose estimation mmpose",
    "sal-hr": "SAL-HR super resolution image restoration",
    "shufflenet-v2": "ShuffleNet V2 efficient CNN mobile architecture",
    "simple-cnn": "convolutional neural network image classification LeNet",
    "slimmable": "slimmable neural networks adjustable width execution",
    "slowfast": "SlowFast networks video recognition dual pathway",
    "sparsebev": "SparseBEV sparse 3D object detection bird eye view",
    "starnet": "StarNet efficient neural network mobile",
    "streampetr": "StreamPETR streaming 3D object detection perception",
    "swin-tiny": "Swin Transformer hierarchical vision shifted window",
    "swin-v2": "Swin Transformer V2 scaled up vision large model",
    "tdl": "TDL text detection localization scene text",
    "timesformer": "TimeSformer divided space-time attention video transformer",
    "unet": "U-Net convolutional biomedical image segmentation",
    "uniformer": "UniFormer unified convolution self-attention vision",
    "v-jepa": "V-JEPA video joint embedding predictive architecture self-supervised",
    "v-jepa-2": "V-JEPA 2 video joint embedding predictive architecture",
    "v-jepa-2-ac": "V-JEPA 2 action conditioned video embedding predictive",
    "vggt-world": "VGGT world geometry transformer 3D reconstruction",
    "videomae": "VideoMAE masked autoencoder video self-supervised",
    "videoprism": "VideoPrism video foundation model encoder",
    "vim": "Vision Mamba state space model visual recognition",
    "vit-adapter": "ViT-Adapter dense prediction vision transformer adapter",
    "vitpose": "ViTPose vision transformer pose estimation",
    "vivit": "ViViT video vision transformer classification",
    "wilor": "WiLoR wild hand pose localization robust",

    # === Graph ===
    "anygraph": "AnyGraph graph foundation model in the wild generalization",
    "cognn": "COGNN cooperative graph neural network message passing",
    "ctne": "CTNE context temporal network embedding dynamic graph",
    "dane": "DANE discriminative network embedding attributed graph",
    "dhne": "DHNE deep hyper-network embedding heterogeneous",
    "dinrl": "inductive network embedding learning transferable node features",
    "dne-pl": "deep network embedding learning partial labeled network",
    "drne": "DRNE deep recursive network embedding role structure",
    "dvne": "DVNE deep variational network embedding distribution",
    "fastne": "FastNE fast network embedding scalable large graph",
    "fmtsf": "financial time series forecasting multi-source graph",
    "gad": "graph anomaly detection multivariate time series sensor",
    "gde": "graph dynamical embedding continuous time dynamic graph",
    "graph2seq": "Graph2Seq graph sequence generation AMR abstract meaning",
    "graphgan": "GraphGAN graph generative adversarial network representation",
    "graphprop": "GraphProp graph representation learning property prediction",
    "mdne": "MDNE multi-dimensional network embedding node representation",
    "mtgnn": "MTGNN multivariate time series graph neural network forecasting",
    "mvcge": "multi-view convolutional graph embedding network representation",
    "mvne": "MVNE multi-view network embedding heterogeneous",
    "nervenet": "NerveNet graph neural network physical simulation",
    "routenet": "RouteNet graph neural network network routing simulation",
    "st-gdn": "spatiotemporal graph deviation network anomaly detection",
    "stemgnn": "STemGNN spatiotemporal graph neural network forecasting",
    "stochastic-gcn": "stochastic training graph convolutional network sampling",
    "structuralseq-gnn": "structural sequence graph neural network embedding",
    "tricl": "TriCL contrastive learning graph neural network tripartite",
    "tpdnr": "TPDNR temporal property dynamic network representation",

    # === Recommendation ===
    "dcn": "Deep Cross Network feature crossing recommendation CTR",
    "dcn-v2": "Deep Cross Network V2 improved feature crossing recommendation",
    "graphprop-rec": "graph property recommendation representation learning",
    "sasrec": "SASRec self-attentive sequential recommendation transformer",
    "sli-rec": "SLi-Rec short long interest recommendation sequential",
    "wide-and-deep": "Wide Deep learning recommender system Google",

    # === Generative ===
    "curiosity-gan": "curiosity generative adversarial network exploration reinforcement",
    "diffusion-unet": "diffusion U-Net latent stable diffusion noise predictor",
    "dit-xl2": "Diffusion Transformer DiT latent patches image generation",
    "gan": "generative adversarial network GAN image generation",
    "lcm": "latent consistency model diffusion fast generation",

    # === Multimodal ===
    "blip2": "BLIP-2 bootstrapping language image pre-training frozen",
    "clip-vit-b32": "CLIP contrastive language image pretraining",
    "siglip-base": "SigLIP sigmoid loss language image pretraining",
    "uni-perceiver": "Uni-Perceiver unified perception multimodal pretraining",

    # === Scientific ===
    "kan": "KAN Kolmogorov Arnold network MLP alternative",
    "kan-2": "KAN Kolmogorov Arnold network science AI symbolic",
    "kan-sr": "KAN symbolic regression Kolmogorov Arnold mathematical",

    # === Reinforcement Learning ===
    "cwm": "conscious world model reinforcement learning neural dynamics",
    "dreamer-v3": "DreamerV3 world model reinforcement learning imagination",
    "rlsf": "RLSF reinforcement learning self-supervised foundation",

    # === Architecture Blocks ===
    "attn-full": "full self-attention transformer dense attention mechanism",
    "attn-sliding-window": "sliding window attention transformer efficient local",
    "attn-sparse": "native sparse attention block efficient long context",
    "mile": "MILE mixture of layers efficient transformer inference",
    "posenc-alibi": "ALiBi attention linear bias position embedding",
    "posenc-learned": "learned absolute position embedding transformer",
    "posenc-rope": "rotary position embedding RoPE transformer",
    "resnet-block": "ResNet residual block identity skip connection",
    "s4": "S4 structured state space sequence model continuous",
    "thnn": "treehouse neural network efficient feedforward",
    "tnif": "TNIF transformer neural inference framework",
    "mile-block": "MILE mixture layers efficient inference transformer",

    # === Time Series ===
    "patch-tst": "PatchTST patch time series transformer forecasting",
    "timesnet": "TimesNet temporal 2D variation modeling time series",
}


def query_openalex(search_query, max_results=10, retries=3):
    """Query OpenAlex API for works matching the search query."""
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
    """Parse OpenAlex API response into list of paper dicts."""
    papers = []
    for work in data.get('results', []):
        # Get title
        title = work.get('title', '') or work.get('display_name', '')
        if not title:
            continue
        title = re.sub(r'\s+', ' ', title.strip())

        # Get DOI or arxiv ID
        doi = work.get('doi', '') or ''
        # Try to extract arxiv ID from various sources
        arxiv_id = ''
        # Check primary_location source
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

        # Also check ids field
        ids = work.get('ids', {})
        if not arxiv_id:
            for key in ('openalex', 'mag', 'doi'):
                val = ids.get(key, '') or ''
                m = re.search(r'(\d{4}\.\d{4,5})', val)
                if m:
                    arxiv_id = m.group(1)
                    break

        # Build ID for dedup: prefer arxiv_id, then DOI, then openalex id
        dedup_id = arxiv_id if arxiv_id else (doi if doi else ids.get('openalex', ''))

        # Get abstract
        abstract = ''
        inv_idx = work.get('abstract_inverted_index', {})
        if inv_idx:
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in inv_idx.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = ' '.join(w for _, w in word_positions)

        # Get authors
        authorships = work.get('authorships', [])
        authors = []
        for a in authorships:
            author = a.get('author', {})
            name = author.get('display_name', '')
            if name:
                authors.append(name)

        # Get publication year
        pub_year = work.get('publication_year', '')

        # Get concepts/keywords
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
    """Get set of arxiv IDs and DOIs already in the papers directory."""
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
            # Extract arxiv IDs
            matches = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content)
            ids.update(matches)
            fn_matches = re.findall(r'(\d{4}\.\d{4,5})', fname)
            ids.update(fn_matches)
            # Extract DOIs
            doi_matches = re.findall(r'doi\.org/(10\.\d+/[^\s`)]+)', content)
            ids.update(doi_matches)
        except:
            pass
    return ids


def get_existing_count(arch):
    """Count existing markdown papers."""
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        return 0
    return len([f for f in os.listdir(papers_path) if f.endswith('.md')])


def slugify(s, maxlen=80):
    """Create a filename-safe slug from a string."""
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:maxlen]


def build_markdown(paper):
    """Build markdown content for a paper."""
    title = paper['title']
    arxiv_id = paper['arxiv_id']
    doi = paper['doi']
    authors = paper['authors']
    summary = paper['summary']
    pub_year = paper['pub_year']
    concepts = paper['concepts']

    # Build source URL
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
    """Query OpenAlex and write papers for an architecture."""
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
    time.sleep(1)  # be polite

    collected = []
    seen = set(existing_ids)

    for paper in results:
        if len(collected) >= need:
            break
        dedup = paper['dedup_id']
        if not dedup:
            # Use title as fallback dedup
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
    arch_list = sorted(ARCH_QUERIES.keys())
    for i, arch in enumerate(arch_list):
        query = ARCH_QUERIES[arch]
        written = process_architecture(arch, query)
        total_written += written

    print(f"\n=== DONE: {total_written} total papers written ===")


if __name__ == '__main__':
    main()
