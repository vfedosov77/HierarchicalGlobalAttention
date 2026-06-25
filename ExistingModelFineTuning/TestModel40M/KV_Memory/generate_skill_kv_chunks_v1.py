#!/usr/bin/env python3
"""
generate_skill_kv_chunks_v2_geo.py

Generate 128 topic-memory chunks, each exactly 64 tokenizer tokens.

Changes from v1:
  - Fixed the topic-count bug: exactly 128 topics.
  - Added a large geography/country section.
  - Removed less useful broad science topics such as quantum computing.
  - Pads short chunks with topic-specific cue tokens, not generic filler.

The generated text is meant only as an initializer for trainable KV-cache memory.
The exact token IDs are the source of truth. Decoded text is a preview only.

Example:
    python generate_skill_kv_chunks_v2_geo.py \
        --tokenizer Qwen/Qwen3-0.6B \
        --output-dir ./skill_memory_geo_out

Outputs:
    skill_chunks.jsonl
    skill_memory_input_ids.json
    skill_memory_input_ids.pt    # if torch is installed
    metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


EXPECTED_TOPICS = 128
Topic = Tuple[str, str, str, str]


TOPICS: List[Topic] = [('Linear algebra',
  'matrix vector eigenvalue rank basis projection',
  'structure and transform vector spaces',
  'reason about embeddings, layers, rotations, projections'),
 ('Calculus',
  'derivative integral gradient limit chain rule',
  'continuous change and accumulation',
  'analyze loss slopes, dynamics, local approximations'),
 ('Probability',
  'random variable distribution expectation Bayes likelihood',
  'uncertainty represented with distributions',
  'estimate chances, priors, evidence, sampling'),
 ('Statistics',
  'mean variance estimator confidence test regression',
  'infer population structure from data',
  'evaluate measurements, noise, significance'),
 ('Information theory',
  'entropy KL mutual information compression coding',
  'quantify information and uncertainty',
  'discuss compression, prediction, token loss'),
 ('Optimization',
  'loss gradient descent step size convergence minima',
  'improve objective by iterative updates',
  'solve training, fitting, search problems'),
 ('Algorithms complexity',
  'big O runtime memory reduction hardness',
  'resource cost of computation',
  'compare scalable algorithms'),
 ('Data structures',
  'array hash heap tree trie queue',
  'organized storage for efficient operations',
  'pick structures for access patterns'),
 ('Graph theory',
  'node edge path tree flow cut centrality',
  'relations represented as networks',
  'solve routing, dependency, topology problems'),
 ('Dynamic programming',
  'state transition recurrence memoization optimal substructure',
  'reuse subproblem solutions',
  'solve sequence, planning, alignment tasks'),
 ('Search planning',
  'A star beam heuristic branch prune',
  'explore possible states toward goals',
  'plan actions under large branching'),
 ('Logic',
  'proposition predicate proof implication satisfiable',
  'formal rules for valid reasoning',
  'check arguments, proofs, constraints'),
 ('Numerical methods',
  'stability precision iteration solver error discretization',
  'approximate math on finite computers',
  'choose stable algorithms and tolerances'),
 ('Distributed systems',
  'consensus replication fault tolerance latency quorum',
  'coordinate computers under failures',
  'design reliable scalable services'),
 ('Operating systems',
  'process thread memory scheduling syscall kernel',
  'manage hardware resources safely',
  'debug performance and concurrency'),
 ('Networking',
  'TCP UDP HTTP routing congestion DNS TLS',
  'communication between machines',
  'reason about protocols and latency'),
 ('Databases',
  'SQL index transaction isolation schema query',
  'persistent structured data management',
  'design storage and retrieval'),
 ('Compilers',
  'parser IR optimization codegen typecheck',
  'translate programs between languages',
  'analyze build, optimization, semantics'),
 ('Programming languages',
  'type system runtime closure generic coroutine',
  'abstractions for expressing computation',
  'compare language features and tradeoffs'),
 ('C++ systems',
  'RAII move template pointer allocator concurrency',
  'high performance native programming',
  'write safe efficient systems code'),
 ('Python tooling',
  'package venv asyncio numpy typing pytest',
  'productive scripting and ML glue',
  'build experiments, utilities, pipelines'),
 ('Java JVM',
  'class bytecode GC thread stream interface',
  'managed object runtime ecosystem',
  'build backend and enterprise services'),
 ('Parallel computing',
  'thread SIMD GPU synchronization race throughput',
  'execute work concurrently',
  'speed up compute-heavy workloads'),
 ('GPU CUDA',
  'kernel warp block shared memory tensor core',
  'massively parallel accelerator programming',
  'optimize deep learning kernels'),
 ('Software architecture',
  'module boundary interface dependency scalability',
  'organize large codebases',
  'choose maintainable designs'),
 ('Testing quality',
  'unit integration property fuzz regression',
  'verify software behavior',
  'catch bugs early'),
 ('Version control CICD',
  'git branch merge review build deploy',
  'coordinate code changes and automation',
  'ship software safely'),
 ('API design',
  'endpoint schema contract version error',
  'define interfaces between systems',
  'create stable usable services'),
 ('Cloud infrastructure',
  'VM storage load balancer autoscale region',
  'run services on remote infrastructure',
  'design scalable deployments'),
 ('Containers Kubernetes',
  'docker image pod service ingress cluster',
  'package and orchestrate services',
  'deploy reproducible applications'),
 ('Cybersecurity',
  'threat vulnerability exploit patch intrusion',
  'protect systems from attacks',
  'assess and mitigate risks'),
 ('Cryptography basics',
  'hash signature encryption key protocol',
  'secure communication and identity',
  'reason about trust and secrecy'),
 ('ML fundamentals',
  'dataset feature label train validation generalization',
  'learn patterns from examples',
  'structure supervised learning experiments'),
 ('Neural networks',
  'layer activation weight neuron hidden representation',
  'composable differentiable function approximators',
  'reason about model architecture'),
 ('Backpropagation',
  'gradient chain rule autograd Jacobian loss',
  'compute parameter updates through computation graphs',
  'debug training signal flow'),
 ('Regularization',
  'dropout weight decay augmentation early stopping',
  'reduce overfitting and improve generalization',
  'stabilize model training'),
 ('Representation learning',
  'embedding latent feature manifold disentangle',
  'learn useful internal coordinates',
  'analyze semantic spaces'),
 ('Transformers',
  'attention MLP residual layer norm token',
  'sequence model using attention and feedforward blocks',
  'discuss LLM architecture'),
 ('Tokenization embeddings',
  'BPE tokenizer vocab embedding subword ids',
  'map text tokens into vectors',
  'handle input representation issues'),
 ('Attention mechanisms',
  'query key value softmax head score context',
  'select information by similarity',
  'explain attention routing and mixing'),
 ('Sparse attention',
  'top k block page query key routing',
  'attend only to important tokens',
  'reduce long context compute'),
 ('KV cache memory',
  'key value cache prefix prefill decode',
  'reuse attention states across generation',
  'store skill memory as trainable cache'),
 ('Positional encoding RoPE',
  'rotary angle position frequency extrapolation',
  'encode token order in attention space',
  'reason about offsets and long context'),
 ('MoE routing',
  'expert gate topk load balance capacity',
  'route tokens to specialized feedforward modules',
  'compare experts with memory pages'),
 ('LoRA adapters',
  'low rank update adapter alpha rank merge',
  'adapt frozen weights with small matrices',
  'compare against trainable KV memory'),
 ('Prompt tuning',
  'prefix virtual token soft prompt adapter',
  'adapt behavior using learned prompt vectors',
  'initialize trainable context'),
 ('Retrieval augmented generation',
  'RAG index embedding search chunk citation',
  'fetch external context for answers',
  'ground knowledge from documents'),
 ('Fine tuning evaluation',
  'SFT dataset loss benchmark ablation',
  'adapt model then measure behavior',
  'compare variants fairly'),
 ('Reinforcement learning',
  'policy reward value action environment',
  'learn decisions from feedback',
  'solve sequential control problems'),
 ('RLHF preference optimization',
  'preference reward model DPO PPO alignment',
  'optimize outputs toward human choices',
  'train assistant behavior'),
 ('Reasoning planning LLMs',
  'chain thought verifier search self consistency',
  'compose steps to solve hard tasks',
  'improve multi-step answers'),
 ('Agents tools',
  'tool call planner memory observation action',
  'LLM controls external functions',
  'build autonomous workflows'),
 ('Interpretability',
  'activation feature circuit attribution probe',
  'understand internal model mechanisms',
  'debug learned representations'),
 ('Uncertainty calibration',
  'confidence probability entropy abstention reliability',
  'match confidence to correctness',
  'decide when to defer'),
 ('Data curation synthetic data',
  'filter deduplicate label generate curriculum',
  'improve training data quality',
  'create useful examples and mixtures'),
 ('Vision CNNs',
  'convolution kernel pooling receptive field',
  'image model using local filters',
  'analyze classic vision networks'),
 ('Vision transformers',
  'patch token image attention ViT',
  'transformer architecture for images',
  'compare CNN and ViT behavior'),
 ('Object detection',
  'bbox anchor proposal YOLO DETR NMS',
  'localize and classify objects',
  'find objects in images'),
 ('Segmentation',
  'mask pixel instance semantic panoptic',
  'assign labels to pixels',
  'separate regions and objects'),
 ('Pose estimation',
  'keypoint skeleton joint heatmap PnP',
  'estimate body or object pose',
  'track geometry and articulation'),
 ('Optical flow',
  'motion vector frame correspondence brightness',
  'estimate pixel motion between frames',
  'reason about video dynamics'),
 ('Stereo depth',
  'disparity baseline triangulation epipolar',
  'recover depth from two cameras',
  'build 3D perception'),
 ('SLAM',
  'localization mapping loop closure bundle adjustment',
  'estimate map and camera trajectory',
  'navigate unknown environments'),
 ('Three dimensional reconstruction',
  'point cloud mesh NeRF SDF triangulation',
  'build 3D shape from observations',
  'model scenes and geometry'),
 ('Multimodal models',
  'image text audio fusion encoder cross attention',
  'combine multiple data modalities',
  'build vision language systems'),
 ('Sensor fusion',
  'IMU camera lidar radar timestamp',
  'combine sensors into robust state',
  'improve robot perception'),
 ('State estimation Kalman',
  'belief observer filter covariance prediction update',
  'estimate hidden variables from measurements',
  'track systems under noise'),
 ('Control theory',
  'feedback stability transfer function controller',
  'choose actions to regulate systems',
  'analyze closed loop behavior'),
 ('PID control',
  'proportional integral derivative setpoint error',
  'simple feedback controller',
  'tune stable practical control'),
 ('Optimal control MPC',
  'trajectory cost constraint horizon control',
  'optimize future actions',
  'plan smooth controlled motion'),
 ('Robot kinematics',
  'joint link transform Jacobian forward inverse',
  'map joint motion to end effector pose',
  'solve robot geometry'),
 ('Motion planning',
  'configuration space obstacle RRT trajectory',
  'find feasible robot motions',
  'avoid collisions'),
 ('Manipulation grasping',
  'gripper object affordance contact grasp',
  'interact physically with objects',
  'plan picking and placing'),
 ('Visual servoing',
  'IBVS PBVS image Jacobian pose error',
  'control robot using visual feedback',
  'connect camera error to velocity'),
 ('Mobile robotics',
  'wheel odometry localization navigation map',
  'robots moving through environments',
  'plan and control ground robots'),
 ('Legged locomotion',
  'gait contact balance terrain footstep',
  'walk or run with legs',
  'control dynamic balance'),
 ('Aerial robotics',
  'drone quadrotor thrust attitude flight',
  'control flying robots',
  'plan stable aerial motion'),
 ('ROS2 middleware',
  'node topic service action launch bag',
  'robot software communication framework',
  'build modular robot systems'),
 ('Sim2real domain randomization',
  'simulation transfer randomization gap adaptation',
  'make simulated policies work in reality',
  'train robust robotic skills'),
 ('Embedded systems',
  'microcontroller firmware interrupt driver memory',
  'software close to hardware',
  'run reliable device code'),
 ('Real time systems',
  'deadline latency jitter scheduler deterministic',
  'compute with timing guarantees',
  'control time-critical systems'),
 ('Edge AI deployment',
  'latency memory power device inference',
  'run ML models on local hardware',
  'optimize deployment constraints'),
 ('Quantization',
  'int8 fp8 scale zero point calibration',
  'reduce model precision efficiently',
  'speed inference and save memory'),
 ('ONNX TensorRT',
  'export graph engine optimize inference',
  'portable optimized model deployment',
  'convert and accelerate networks'),
 ('MLOps monitoring',
  'pipeline drift logging metrics rollback',
  'operate ML systems in production',
  'track data and model health'),
 ('Experiment tracking',
  'run config seed metric artifact ablation',
  'record reproducible experiments',
  'compare training variants'),
 ('Evaluation benchmarks',
  'metric testset leaderboard contamination robustness',
  'measure model quality systematically',
  'choose fair evaluation protocols'),
 ('Project specific memory',
  'user project skill KV cache robot AI startup',
  'private evolving project context',
  'store task-specific facts after training'),
 ('World geography',
  'continent ocean latitude longitude capital border river mountain',
  'global places organized by location',
  'route questions about maps countries cities regions'),
 ('Europe geography',
  'Europe EU Schengen Alps Baltic Balkans Mediterranean capitals',
  'European countries cities regions and borders',
  'answer questions about European places'),
 ('North America geography',
  'North America USA Canada Mexico Caribbean cities states provinces',
  'places across northern American continent',
  'route regional city country questions'),
 ('South America geography',
  'South America Andes Amazon Brazil Argentina Chile Peru Colombia',
  'countries cities rivers mountains of South America',
  'answer regional geography questions'),
 ('Asia geography',
  'Asia China India Japan Korea Southeast Central Asia cities',
  'large Asian regions countries capitals and megacities',
  'route Asian place knowledge'),
 ('Africa geography',
  'Africa Sahara Nile Maghreb Sahel Lagos Cairo Johannesburg',
  'African regions countries cities rivers deserts',
  'answer Africa place questions'),
 ('Middle East geography',
  'Middle East Arabia Levant Gulf Iran Turkey Egypt cities',
  'countries and cities around West Asia and North Africa',
  'route Middle East knowledge'),
 ('Oceania geography',
  'Oceania Australia New Zealand Pacific islands Sydney Auckland',
  'Pacific region countries islands and cities',
  'answer Oceania place questions'),
 ('United States',
  'USA United States America Washington DC New York Los Angeles Chicago Houston',
  'federal country with states major cities coasts rivers',
  'answer US geography cities states institutions'),
 ('Canada',
  'Canada Ottawa Toronto Montreal Vancouver Calgary Quebec Ontario British Columbia',
  'large North American country with provinces and major cities',
  'answer Canada geography and city questions'),
 ('Mexico',
  'Mexico Mexico City Guadalajara Monterrey Puebla Cancun Yucatan Baja',
  'North American country with states coasts and major cities',
  'answer Mexico geography culture travel questions'),
 ('Germany',
  'Germany Berlin Hamburg Munich Cologne Frankfurt Bavaria Rhine Ruhr Saxony',
  'central European federal country with states and major cities',
  'answer Germany geography cities universities industry'),
 ('France',
  'France Paris Marseille Lyon Toulouse Nice Bordeaux Alps Loire Seine',
  'western European country with regions cities rivers coasts',
  'answer France geography cities culture'),
 ('United Kingdom',
  'United Kingdom UK England Scotland Wales Northern Ireland London Manchester Edinburgh',
  'island state with constituent countries and major cities',
  'answer UK geography and city questions'),
 ('Spain',
  'Spain Madrid Barcelona Valencia Seville Bilbao Malaga Andalusia Catalonia',
  'Iberian country with regions islands and major cities',
  'answer Spain geography cities culture'),
 ('Italy',
  'Italy Rome Milan Naples Turin Florence Venice Sicily Sardinia',
  'peninsula country with regions islands and historic cities',
  'answer Italy geography and city questions'),
 ('Netherlands',
  'Netherlands Amsterdam Rotterdam The Hague Utrecht Eindhoven Randstad Holland',
  'low country with provinces ports canals and cities',
  'answer Dutch geography and city questions'),
 ('Belgium',
  'Belgium Brussels Antwerp Ghent Bruges Wallonia Flanders Leuven',
  'European country with regions languages and cities',
  'answer Belgium geography and institutional questions'),
 ('Switzerland',
  'Switzerland Bern Zurich Geneva Basel Lausanne Alps cantons',
  'Alpine federal country with cantons and multilingual cities',
  'answer Swiss geography cities institutions'),
 ('Austria',
  'Austria Vienna Graz Linz Salzburg Innsbruck Alps Danube',
  'central European Alpine country with states and cities',
  'answer Austria geography and city questions'),
 ('Poland',
  'Poland Warsaw Krakow Wroclaw Gdansk Poznan Vistula Silesia',
  'central European country with regions rivers and cities',
  'answer Poland geography and city questions'),
 ('Czechia',
  'Czechia Czech Republic Prague Brno Ostrava Plzen Bohemia Moravia',
  'central European country with historic regions and cities',
  'answer Czech geography and city questions'),
 ('Russia',
  'Russia Moscow Saint Petersburg Novosibirsk Siberia Volga Ural Far East',
  'large Eurasian country with regions rivers and cities',
  'answer Russia geography and city questions'),
 ('Ukraine',
  'Ukraine Kyiv Kharkiv Odesa Lviv Dnipro Crimea Carpathians',
  'eastern European country with regions rivers and cities',
  'answer Ukraine geography and city questions'),
 ('Turkey',
  'Turkey Ankara Istanbul Izmir Antalya Anatolia Bosporus Black Sea',
  'country connecting Europe and Asia with major cities',
  'answer Turkey geography and city questions'),
 ('China',
  'China Beijing Shanghai Guangzhou Shenzhen Chengdu Wuhan Pearl Yangtze',
  'large East Asian country with provinces megacities rivers',
  'answer China geography and city questions'),
 ('Japan',
  'Japan Tokyo Osaka Kyoto Yokohama Hokkaido Honshu Kyushu Shikoku',
  'island country with regions prefectures and cities',
  'answer Japan geography and city questions'),
 ('South Korea',
  'South Korea Seoul Busan Incheon Daegu Daejeon Jeju Han',
  'East Asian country with provinces cities and peninsula geography',
  'answer Korea geography and city questions'),
 ('India',
  'India New Delhi Mumbai Bengaluru Chennai Kolkata Hyderabad Ganges Deccan',
  'South Asian country with states megacities rivers regions',
  'answer India geography and city questions'),
 ('Indonesia',
  'Indonesia Jakarta Surabaya Bandung Bali Java Sumatra Borneo Sulawesi',
  'archipelago country with islands provinces and major cities',
  'answer Indonesia geography and city questions'),
 ('Australia',
  'Australia Canberra Sydney Melbourne Brisbane Perth Adelaide Tasmania Outback',
  'continent country with states coasts deserts and cities',
  'answer Australia geography and city questions'),
 ('Brazil',
  'Brazil Brasilia Sao Paulo Rio de Janeiro Salvador Amazon Bahia',
  'South American country with states rivers rainforest cities',
  'answer Brazil geography and city questions'),
 ('Argentina',
  'Argentina Buenos Aires Cordoba Rosario Mendoza Patagonia Pampas Andes',
  'South American country with provinces regions and cities',
  'answer Argentina geography and city questions'),
 ('Chile',
  'Chile Santiago Valparaiso Atacama Patagonia Andes Pacific Antofagasta',
  'long Pacific country with deserts mountains and cities',
  'answer Chile geography and city questions'),
 ('Egypt',
  'Egypt Cairo Alexandria Giza Nile Sinai Luxor Aswan Delta',
  'North African country with Nile valley desert and cities',
  'answer Egypt geography and city questions'),
 ('South Africa',
  'South Africa Pretoria Cape Town Johannesburg Durban Gauteng Western Cape',
  'southern African country with provinces coasts and cities',
  'answer South Africa geography city questions'),
 ('Nigeria',
  'Nigeria Abuja Lagos Kano Ibadan Port Harcourt Niger Delta',
  'West African country with states rivers and major cities',
  'answer Nigeria geography and city questions'),
 ('Saudi Arabia',
  'Saudi Arabia Riyadh Jeddah Mecca Medina Dammam Hijaz Neom',
  'Arabian country with regions holy cities coasts desert',
  'answer Saudi geography and city questions'),
 ('United Arab Emirates',
  'UAE Abu Dhabi Dubai Sharjah Ajman Fujairah Gulf emirates',
  'Gulf federation with emirates cities ports and desert',
  'answer UAE geography and city questions'),
 ('Israel',
  'Israel Jerusalem Tel Aviv Haifa Beersheba Galilee Negev Mediterranean',
  'Middle Eastern country with coastal plains hills desert cities',
  'answer Israel geography and city questions')]


def make_topic_text(index: int, topic: Topic) -> str:
    name, cues, core, use = topic
    return (
        f"Topic {index:03d}: {name}. "
        f"Cues: {cues}. "
        f"Core: {core}. "
        f"Use: {use}."
    )


def make_topic_filler(topic: Topic) -> str:
    name, cues, _, _ = topic
    # Padding should reinforce the topic, because these tokens will become
    # attention-addressable memory. Keep it compact and semantic.
    return f" ; {name} {cues}"


def encode_exact(
    tokenizer,
    text: str,
    target_len: int,
    filler_text: str,
) -> Tuple[List[int], bool, int, int]:
    """Return exactly target_len token IDs.

    If text is too long, truncate. If it is too short, append cyclic topic-specific
    filler IDs. Returns:
        ids, was_truncated, original_len, filler_len
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    original_len = len(ids)

    if len(ids) > target_len:
        return ids[:target_len], True, original_len, 0

    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("topic filler tokenized to an empty list")

    filler_used = 0
    if len(ids) < target_len:
        need = target_len - len(ids)
        repeats = (need + len(filler_ids) - 1) // len(filler_ids)
        ids = ids + (filler_ids * repeats)[:need]
        filler_used = need

    return ids, False, original_len, filler_used


def validate_topics() -> None:
    if len(TOPICS) != EXPECTED_TOPICS:
        names = "\n".join(f"{i:03d} {t[0]}" for i, t in enumerate(TOPICS))
        raise ValueError(
            f"Expected {EXPECTED_TOPICS} topics, got {len(TOPICS)}.\n"
            f"Current topics:\n{names}"
        )

    seen = set()
    duplicates = []
    for topic in TOPICS:
        if topic[0] in seen:
            duplicates.append(topic[0])
        seen.add(topic[0])
    if duplicates:
        raise ValueError(f"Duplicate topic names: {duplicates}")


def build_chunks(
    tokenizer_name: str,
    tokens_per_topic: int,
    output_dir: Path,
    trust_remote_code: bool,
) -> None:
    validate_topics()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    records: List[Dict] = []
    all_ids: List[int] = []
    truncated_count = 0
    total_filler_tokens = 0

    for i, topic in enumerate(TOPICS):
        raw_text = make_topic_text(i, topic)
        filler_text = make_topic_filler(topic)

        ids, was_truncated, original_len, filler_used = encode_exact(
            tokenizer=tokenizer,
            text=raw_text,
            target_len=tokens_per_topic,
            filler_text=filler_text,
        )

        decoded_preview = tokenizer.decode(ids, clean_up_tokenization_spaces=False)

        if was_truncated:
            truncated_count += 1
        total_filler_tokens += filler_used

        all_ids.extend(ids)
        records.append(
            {
                "topic_id": i,
                "topic_name": topic[0],
                "token_count": len(ids),
                "original_token_count": original_len,
                "filler_tokens_used": filler_used,
                "was_truncated": was_truncated,
                "text_before_padding_or_truncation": raw_text,
                "topic_specific_filler": filler_text,
                "decoded_exact_token_preview": decoded_preview,
                "token_ids": ids,
            }
        )

    expected_total = len(TOPICS) * tokens_per_topic
    if len(all_ids) != expected_total:
        raise RuntimeError(f"Bad total length: got {len(all_ids)}, expected {expected_total}")

    jsonl_path = output_dir / "skill_chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ids_path = output_dir / "skill_memory_input_ids.json"
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    metadata = {
        "tokenizer": tokenizer_name,
        "num_topics": len(TOPICS),
        "tokens_per_topic": tokens_per_topic,
        "total_tokens": len(all_ids),
        "truncated_chunks": truncated_count,
        "total_filler_tokens": total_filler_tokens,
        "geography_topic_range": [88, 127],
        "note": (
            "Use skill_memory_input_ids.json or .pt as the exact initializer. "
            "Decoded text is only a preview; tokenizer round-trip is not guaranteed."
        ),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    torch_path = output_dir / "skill_memory_input_ids.pt"
    try:
        import torch

        tensor = torch.tensor(all_ids, dtype=torch.long).view(len(TOPICS), tokens_per_topic)
        torch.save(tensor, torch_path)
        saved_torch = True
    except Exception:
        saved_torch = False

    print(f"Saved: {jsonl_path}")
    print(f"Saved: {ids_path}")
    print(f"Saved: {metadata_path}")
    if saved_torch:
        print(f"Saved: {torch_path}")
    else:
        print("Skipped .pt save because torch is not available.")
    print(f"Total tokens: {len(all_ids)} = {len(TOPICS)} x {tokens_per_topic}")
    print(f"Truncated chunks: {truncated_count}")
    print(f"Topic-specific filler tokens used: {total_filler_tokens}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face tokenizer name or local path.",
    )
    parser.add_argument(
        "--tokens-per-topic",
        type=int,
        default=64,
        help="Exact number of tokenizer tokens per topic block.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("skill_memory_geo_out"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code in AutoTokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_chunks(
        tokenizer_name=args.tokenizer,
        tokens_per_topic=args.tokens_per_topic,
        output_dir=args.output_dir,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()