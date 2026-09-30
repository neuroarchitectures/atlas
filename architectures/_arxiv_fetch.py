#!/usr/bin/env python3
"""
Query arXiv API for related papers for each LM architecture and convert
results into markdown files in each references/papers/ directory.

Target: 2-4 papers per architecture (beyond the primary paper already present).
"""

import os
import re
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

BASE = "/Users/xiaming/Workspace/atlas/architectures"

# Mapping: architecture -> list of search queries (will try each until we get enough papers)
# Each query uses arXiv API field prefixes: ti (title), abs (abstract), all (all fields)
ARCH_QUERIES = {
    # === LLM families ===
    "llama3-8b": [
        'ti:"Llama 3"',
        'ti:RoPE AND ti:attention AND ti:grouped',
        'ti:SwiGLU AND abs:language model',
        'ti:RMSNorm AND abs:transformer',
    ],
    "llama3-block": [
        'ti:"grouped query attention"',
        'ti:RoPE AND ti:rotary AND ti:embedding',
        'ti:SwiGLU AND abs:feed-forward',
        'ti:"rotary position embedding"',
    ],
    "llama-4-scout": [
        'ti:"Llama 4"',
        'ti:"mixture of experts" AND ti:attention',
        'ti:"sparse mixture" AND abs:language model',
        'ti:MoE AND ti:vision AND abs:language',
    ],
    "mistral-7b": [
        'ti:Mistral',
        'ti:"sliding window attention"',
        'ti:"grouped query attention"',
        'ti:"rotary embedding" AND abs:language model',
    ],
    "mixtral-block": [
        'ti:"Mixtral"',
        'ti:"sparse mixture of experts"',
        'ti:"mixture of experts" AND ti:transformer AND abs:language',
        'ti:MoE AND ti:routing AND abs:language model',
    ],
    "qwen2.5-7b": [
        'ti:Qwen',
        'ti:"grouped query attention" AND abs:language',
        'ti:"rotary position embedding" AND abs:transformer',
        'ti:"SwiGLU" AND abs:language model',
    ],
    "qwen3-8b": [
        'ti:Qwen3',
        'ti:"grouped query attention" AND abs:language',
        'ti:"mixture of experts" AND ti:transformer',
        'ti:"sliding window" AND abs:language model',
    ],
    "deepseek-v2": [
        'ti:"DeepSeek-V2"',
        'ti:"multi-head latent attention"',
        'ti:"mixture of experts" AND abs:language AND ti:DeepSeek',
        'ti:"latent attention" AND abs:transformer',
    ],
    "deepseek-v3": [
        'ti:"DeepSeek-V3"',
        'ti:"multi-head latent attention"',
        'ti:"mixture of experts" AND abs:language AND ti:DeepSeek',
        'ti:"auxiliary loss free" AND abs:language model',
    ],
    "deepseek-v2-lite": [
        'ti:"DeepSeek-V2"',
        'ti:"multi-head latent attention"',
        'ti:"mixture of experts" AND ti:DeepSeek',
        'ti:"latent attention" AND abs:transformer',
    ],
    "deepseek-llm-7b": [
        'ti:"DeepSeek LLM"',
        'ti:DeepSeek AND abs:language model',
        'ti:"pre-training" AND abs:language model AND ti:DeepSeek',
        'ti:"grouped query attention" AND abs:language model',
    ],
    "deepseekmath": [
        'ti:"DeepSeekMath"',
        'ti:mathematical AND ti:reasoning AND abs:language model',
        'ti:"grouped relative attention"',
        'ti:math AND ti:reasoning AND abs:transformer',
    ],
    "gpt-1": [
        'ti:"generative pre-training" AND abs:language',
        'ti:"improving language understanding" AND abs:transformer',
        'ti:"pre-training" AND abs:language model AND ti:Radford',
        'ti:"decoder only" AND abs:transformer AND abs:language',
    ],
    "gpt-2": [
        'ti:"Language Models are Unsupervised Multitask Learners"',
        'ti:"zero-shot" AND abs:language model AND ti:transformer',
        'ti:"pre-training" AND abs:language model AND ti:Radford',
        'ti:"decoder only" AND abs:language model',
    ],
    "gpt-3": [
        'ti:"Language Models are Few-Shot Learners"',
        'ti:"in-context learning" AND abs:language model',
        'ti:"few-shot" AND abs:language model AND ti:Brown',
        'ti:"scaling laws" AND abs:language model',
    ],
    "gpt-4": [
        'ti:"GPT-4"',
        'ti:"multimodal" AND abs:language model AND ti:RLHF',
        'ti:"reinforcement learning" AND abs:human feedback AND abs:language',
        'ti:"predictable scaling" AND abs:language model',
    ],
    "gpt-oss-20b": [
        'ti:"gpt-oss"',
        'ti:"open source" AND abs:language model AND ti:OpenAI',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"mixture of experts" AND abs:language AND ti:open',
    ],
    "gpt-oss-120b": [
        'ti:"gpt-oss"',
        'ti:"mixture of experts" AND abs:language model AND ti:OpenAI',
        'ti:"open source" AND abs:language model',
        'ti:"sparse attention" AND abs:language model',
    ],
    "gemini": [
        'ti:"Gemini"',
        'ti:multimodal AND abs:language model AND ti:Google',
        'ti:"multimodal" AND abs:reasoning',
        'ti:"native multimodal" AND abs:language',
    ],
    "glm-4.5-air": [
        'ti:"GLM-4"',
        'ti:"GLM" AND abs:language model AND ti:Zhipu',
        'ti:"autoregressive blank filling" AND abs:language',
        'ti:"grouped query attention" AND abs:language model AND ti:GLM',
    ],
    "gemma-4-12b": [
        'ti:"Gemma"',
        'ti:Gemma AND abs:language model AND ti:Google',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"rotary embedding" AND abs:language model',
    ],
    "phi-2": [
        'ti:"Phi-2"',
        'ti:"textbooks are all you need"',
        'ti:"small language model" AND abs:reasoning',
        'ti:"knowledge" AND abs:language model AND ti:Microsoft',
    ],
    "phi3-mini": [
        'ti:"Phi-3"',
        'ti:"small language model" AND abs:mobile',
        'ti:"rotary position embedding" AND abs:language model',
        'ti:"knowledge" AND abs:language model AND ti:Microsoft',
    ],
    "yi-6b": [
        'ti:"Yi" AND abs:language model',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"rotary position embedding" AND abs:language model',
        'ti:"bilingual" AND abs:language model',
    ],
    "baichuan2-7b": [
        'ti:"Baichuan"',
        'ti:"bilingual" AND abs:language model AND ti:Chinese',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"rotary position embedding" AND abs:language model',
    ],
    "chatglm3-6b": [
        'ti:"ChatGLM"',
        'ti:"GLM" AND abs:language model AND ti:chat',
        'ti:"autoregressive blank filling" AND abs:language',
        'ti:"bilingual" AND abs:language model AND ti:Chinese',
    ],
    "internlm2-7b": [
        'ti:"InternLM"',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"rotary position embedding" AND abs:language model',
        'ti:"long context" AND abs:language model',
    ],
    "mpt-7b": [
        'ti:"MPT" AND abs:language model',
        'ti:"MosaicML" AND abs:language model',
        'ti:"flash attention" AND abs:language model',
        'ti:"ALiBi" AND abs:language model',
    ],
    "falcon-7b": [
        'ti:"Falcon" AND abs:language model',
        'ti:"rotary" AND abs:language model AND ti:Falcon',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"multi query attention" AND abs:language model',
    ],
    "olmo-7b": [
        'ti:"OLMo"',
        'ti:"open language model" AND abs:Allen',
        'ti:"open source" AND abs:language model AND ti:training',
        'ti:"grouped query attention" AND abs:language model',
    ],
    "minicpm-2b": [
        'ti:"MiniCPM"',
        'ti:"small language model" AND abs:edge',
        'ti:"knowledge" AND abs:language model AND ti:efficient',
        'ti:"rotary position embedding" AND abs:language model',
    ],
    "pythia-1.4b": [
        'ti:"Pythia"',
        'ti:"EleutherAI" AND abs:language model',
        'ti:"scaling" AND abs:language model AND ti:analysis',
        'ti:"Pile" AND abs:language model',
    ],
    "skywork-13b": [
        'ti:"Skywork"',
        'ti:"bilingual" AND abs:language model AND ti:Chinese',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"pre-training" AND abs:language model AND ti:scaling',
    ],
    "nemotron": [
        'ti:"Nemotron"',
        'ti:NVIDIA AND abs:language model',
        'ti:"grouped query attention" AND abs:language model',
        'ti:"inference" AND abs:language model AND ti:efficient',
    ],
    "jamba": [
        'ti:"Jamba"',
        'ti:"hybrid" AND abs:transformer AND ti:SSM',
        'ti:"state space model" AND ti:transformer AND abs:language',
        'ti:"mixture of experts" AND abs:language model AND ti:hybrid',
    ],
    "kimi-k2.6": [
        'ti:"Kimi"',
        'ti:"Moonshot" AND abs:language model',
        'ti:"multi-head latent attention" AND abs:language model',
        'ti:"mixture of experts" AND abs:language model AND ti:long context',
    ],
    "semi-gpt-4": [
        'ti:"semi-supervised" AND abs:language model',
        'ti:"GPT-4" AND abs:semi-supervised',
        'ti:"unlabeled data" AND abs:language model',
        'ti:"self-training" AND abs:language model',
    ],
    "ngpt": [
        'ti:"nGPT"',
        'ti:"normalized transformer"',
        'ti:"unit norm" AND abs:transformer AND abs:language',
        'ti:"representation" AND abs:transformer AND ti:normalization',
    ],
    # === SSMs ===
    "mamba": [
        'ti:"Mamba"',
        'ti:"selective state space"',
        'ti:"state space model" AND abs:language AND ti:sequence',
        'ti:"linear time" AND abs:sequence model AND ti:state space',
    ],
    "mamba-3": [
        'ti:"Mamba-3"',
        'ti:"trapezoidal" AND abs:state space model',
        'ti:"complex" AND abs:state space model AND ti:SSM',
        'ti:"MIMO" AND abs:state space model',
    ],
    "mambavision": [
        'ti:"MambaVision"',
        'ti:"Mamba" AND ti:vision AND abs:backbone',
        'ti:"hybrid" AND ti:Mamba AND ti:transformer AND abs:vision',
        'ti:"state space" AND ti:vision AND abs:image',
    ],
    "localmamba": [
        'ti:"LocalMamba"',
        'ti:"local scanning" AND abs:Mamba',
        'ti:"windowed" AND abs:Mamba AND abs:vision',
        'ti:"scan" AND abs:state space model AND abs:vision',
    ],
    "vmamba": [
        'ti:"VMamba"',
        'ti:"visual state space"',
        'ti:"2D selective scan" AND abs:vision',
        'ti:"state space" AND ti:vision AND abs:backbone',
    ],
    "rwkv": [
        'ti:"RWKV"',
        'ti:"receptance weighted key value"',
        'ti:"linear attention" AND abs:language model AND ti:recurrent',
        'ti:"channel wise" AND abs:time decay AND abs:language',
    ],
    "xlstm": [
        'ti:"xLSTM"',
        'ti:"extended LSTM"',
        'ti:"exponential gating" AND abs:LSTM',
        'ti:"matrix memory" AND abs:LSTM',
    ],
    # === Core architectures ===
    "transformer": [
        'ti:"Attention is All You Need"',
        'ti:"scaled dot product attention"',
        'ti:"multi head attention" AND abs:transformer',
        'ti:"self attention" AND abs:transformer AND abs:parallel',
    ],
    "differential-transformer": [
        'ti:"Differential Transformer"',
        'ti:"differential attention" AND abs:transformer',
        'ti:"attention" AND abs:noise AND abs:cancel AND abs:transformer',
        'ti:"attention" AND abs:subtraction AND abs:transformer',
    ],
    "modernbert": [
        'ti:"ModernBERT"',
        'ti:"Modern BERT"',
        'ti:"encoder" AND abs:BERT AND ti:modern',
        'ti:"rotary" AND abs:BERT AND ti:efficient',
    ],
    "bert-base": [
        'ti:"BERT"',
        'ti:"bidirectional encoder representations"',
        'ti:"masked language model" AND abs:transformer',
        'ti:"pre-training" AND abs:language AND ti:Devlin',
    ],
    "bert4rec": [
        'ti:"BERT4Rec"',
        'ti:"bidirectional" AND abs:recommendation AND ti:transformer',
        'ti:"sequential recommendation" AND abs:BERT',
        'ti:"masked" AND abs:recommendation AND ti:transformer',
    ],
    "t5-small": [
        'ti:"T5"',
        'ti:"text-to-text transfer transformer"',
        'ti:"encoder decoder" AND abs:transformer AND abs:language',
        'ti:"unified" AND abs:language model AND ti:text',
    ],
    "backpack-lm": [
        'ti:"Backpack"',
        'ti:"sense embeddings" AND abs:language model',
        'ti:"interpretable" AND abs:language model AND ti:contextualization',
        'ti:"nonlinear" AND abs:language model AND ti:embedding',
    ],
    "cldlm": [
        'ti:"contrastive learning" AND abs:dual AND abs:linear',
        'ti:"contrastive" AND abs:representation learning AND ti:linear',
        'ti:"dual mapping" AND abs:contrastive',
        'ti:"contrastive" AND abs:representation AND ti:learning',
    ],
    "megabyte": [
        'ti:"Megabyte"',
        'ti:"multi scale" AND abs:byte AND abs:language model',
        'ti:"sub quadratic" AND abs:language model AND ti:byte',
        'ti:"patch" AND abs:byte AND abs:language model',
    ],
    # === Audio ===
    "encodec": [
        'ti:"EnCodec"',
        'ti:"neural audio codec"',
        'ti:"audio compression" AND abs:neural',
        'ti:"residual vector quantization" AND abs:audio',
    ],
    "whisper-small": [
        'ti:"Whisper"',
        'ti:"robust speech recognition" AND abs:multilingual',
        'ti:"weakly supervised" AND abs:speech recognition',
        'ti:"audio" AND abs:transcription AND abs:multilingual',
    ],
    "wav2vec2-base": [
        'ti:"wav2vec 2.0"',
        'ti:"self supervised" AND abs:speech AND ti:contrastive',
        'ti:"speech representation" AND abs:self supervised',
        'ti:"wav2vec" AND abs:learning AND abs:speech',
    ],
    "hubert-base": [
        'ti:"HuBERT"',
        'ti:"hidden unit" AND abs:speech AND ti:representation',
        'ti:"self supervised" AND abs:speech AND ti:clustering',
        'ti:"speech representation" AND abs:self supervised AND ti:BERT',
    ],
    # === Multimodal ===
    "flamingo": [
        'ti:"Flamingo"',
        'ti:"visual language model" AND abs:few shot',
        'ti:"cross attention" AND abs:vision AND abs:language',
        'ti:"perceiver" AND abs:visual AND abs:language',
    ],
    "blip2": [
        'ti:"BLIP-2"',
        'ti:"bootstrapping" AND abs:vision language AND ti:pre-training',
        'ti:"Q-Former" AND abs:vision AND abs:language',
        'ti:"frozen" AND abs:language model AND ti:vision',
    ],
    "llava-1.5-7b": [
        'ti:"LLaVA"',
        'ti:"visual instruction tuning"',
        'ti:"multimodal" AND abs:language model AND ti:vision',
        'ti:"instruction tuning" AND abs:vision AND abs:language',
    ],
    "instructgpt": [
        'ti:"InstructGPT"',
        'ti:"training language models" AND abs:instructions AND ti:human',
        'ti:"RLHF" AND abs:language model AND ti:alignment',
        'ti:"reinforcement learning" AND abs:human feedback AND abs:language',
    ],
}


def query_arxiv(query, max_results=5):
    """Query arXiv API and return list of paper dicts."""
    search_url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}"
    )
    req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req, timeout=30)
        xml_data = response.read().decode('utf-8')
    except Exception as e:
        print(f"  ERROR querying arxiv: {e}")
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        for entry in root.findall('atom:entry', ns):
            entry_id = entry.find('atom:id', ns)
            if entry_id is None:
                continue
            arxiv_url = entry_id.text.strip()
            # Extract arxiv ID from URL like http://arxiv.org/abs/2304.06975v1
            id_match = re.search(r'abs/(\d+\.\d+)', arxiv_url)
            if not id_match:
                continue
            arxiv_id = id_match.group(1)

            title_el = entry.find('atom:title', ns)
            title = title_el.text.strip() if title_el is not None and title_el.text else ''
            # Clean up whitespace
            title = re.sub(r'\s+', ' ', title)

            summary_el = entry.find('atom:summary', ns)
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ''
            summary = re.sub(r'\s+', ' ', summary)

            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_el = author.find('atom:name', ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # Published date
            pub_el = entry.find('atom:published', ns)
            published = pub_el.text.strip() if pub_el is not None and pub_el.text else ''

            # Categories
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term', '')
                if term:
                    categories.append(term)

            # PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            papers.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': summary,
                'authors': authors,
                'published': published,
                'categories': categories,
                'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                'pdf_url': pdf_url,
            })
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
    return papers


def get_existing_arxiv_ids(arch):
    """Get set of arxiv IDs already in the papers directory."""
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
            fn_matches = re.findall(r'(\d{4}\.\d+)', fname)
            ids.update(fn_matches)
        except:
            pass
    return ids


def sanitize_filename(title, arxiv_id):
    """Create a clean filename from paper title and arxiv ID."""
    # Remove special chars, replace spaces with underscores
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())[:80]
    return f"{clean}_{arxiv_id}.md"


def create_paper_md(paper, arch):
    """Create markdown content for a paper."""
    authors_str = ", ".join(paper['authors'][:10])
    if len(paper['authors']) > 10:
        authors_str += " et al."

    year = ""
    if paper['published']:
        year = paper['published'][:4]

    # Determine a label for the first author/org
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


def process_architecture(arch, queries, target_count=3):
    """Query arXiv for related papers and write markdown files."""
    papers_path = os.path.join(BASE, arch, "references", "papers")
    if not os.path.isdir(papers_path):
        print(f"  {arch}: papers dir missing, skipping")
        return 0

    existing_ids = get_existing_arxiv_ids(arch)
    print(f"  {arch}: existing IDs = {existing_ids}")

    collected = []
    seen_ids = set(existing_ids)

    for query in queries:
        if len(collected) >= target_count:
            break
        print(f"    Query: {query}")
        results = query_arxiv(query, max_results=5)
        time.sleep(3.5)  # Respect arXiv rate limit (1 request per 3 sec)

        for paper in results:
            if len(collected) >= target_count:
                break
            if paper['arxiv_id'] in seen_ids:
                continue
            # Skip very short or empty titles
            if len(paper['title']) < 10:
                continue
            seen_ids.add(paper['arxiv_id'])
            collected.append(paper)
            print(f"    Found: {paper['arxiv_id']} - {paper['title'][:60]}")

    # Write markdown files
    written = 0
    for paper in collected:
        md_content = create_paper_md(paper, arch)
        filename = sanitize_filename(paper['title'], paper['arxiv_id'])
        fpath = os.path.join(papers_path, filename)
        # Avoid overwriting existing files
        if os.path.exists(fpath):
            filename = f"Related_{paper['arxiv_id']}.md"
            fpath = os.path.join(papers_path, filename)
        with open(fpath, 'w') as f:
            f.write(md_content)
        written += 1
        print(f"    Written: {filename}")

    return written


def main():
    total_written = 0
    for arch, queries in ARCH_QUERIES.items():
        print(f"\n=== Processing: {arch} ===")
        written = process_architecture(arch, queries, target_count=3)
        total_written += written
        print(f"  {arch}: {written} papers written")

    print(f"\n=== DONE: {total_written} total papers written ===")


if __name__ == '__main__':
    main()
