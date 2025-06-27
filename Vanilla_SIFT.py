# 13_en_bpb.py  – activeft + VTL で λ を振りながら
#   ・Pre-FT / Post-FT の bits-per-byte (bpb) を測定
#   ・各ステップの loss と bpb を同時表示
#
# pip install "activeft[faiss]" bitsandbytes peft "transformers>=4.41"

import math, copy, torch, faiss, numpy as np, bitsandbytes as bnb
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from activeft.sift import Retriever
from activeft.acquisition_functions.vtl import VTL

# ---------- tiny corpus ----------
RAW_PARAS = [
    "京都大学人工知能研究会 KaiRA は、AI・機械学習の専門書を輪読する自主ゼミ形式のサークルである。",
    "文理・大学を問わず多様なバックグラウンドの学生が参加している。",
    "AI を学びたい学生や一緒に AI を開発したい学生を歓迎している。",
    "書籍購入費や計算リソース費用の補助がある。",
    "活動の中心は輪読会と作業会である。",
    "興味のある本があればメンバー主導で勉強会を立ち上げられる。",
    "論文読み会や Kaggle・SIGNATE コンペへのチーム参加も行っている。",
    "会費は無料である。",
    "輪読会は毎週月曜・木曜の 18:40 から京都大学文学部教室とオンラインで開催される。",
    "コンペ練習会は毎週土曜 10:00 から行われる。",
    "論文読み会は月に一度、約 2 時間実施される。",
    "長期休暇と試験期間中は定期活動を休止する。",
    "KaiRA は毎年 11 月祭でポスター展示・AI デモ・会誌販売を行う。",
    "会長は工学部理工化学科の岡本和優である。",
    "副会長は理学部数理科学系の千葉一世である。",
    "会計は工学部情報学科の宮前明生である。",
    "広報は工学部情報学科の稲葉陽孔である。",
    "技術顧問には情報学研究科の鹿島久嗣教授が就任している。",
    "小島諒介准教授は医療データの時系列解析を専門とする。",
    "KaiRA は株式会社 Rist と株式会社スクラムサインから支援を受けている。",
    "創設者の金子英樹は、難解な数式に挫折する学生のために KaiRA を立ち上げた。",
    "歴代会長には金子英樹、大山百々勢、三宅大貴、松田拓巳が名を連ねる。",
    "松田拓巳は深層学習を用いた気象アルゴリズム研究に従事している。",
    "入会ステップ 1 は connpass で輪読会に申し込み見学することである。",
    "3〜4 月には新入会希望者向け説明会を実施し、最新情報は X で告知される。",
    "入会ステップ 2 は connpass の入会申請フォームに回答し、確認後 Slack 招待を受ける。",
    "会費は 0 円で、発表者には書籍代、Kaggle 参加者には計算リソース代が補助される。",
    "プログラミング初心者も参加可能で、Python を並行学習すると理解が深まる。",
    "他大学の学生もオンライン参加が可能である。",
    "事前学習には『ゼロから作る Deep Learning』の読書が推奨される。",
    "メンバーは学んだことを活かすためにソフトウェアプロジェクトの開発にも取り組んでいます。",
    "定期的な活動は大学の長期休暇期間や試験期間中は休止します。",
    "開催教室は文学部の空き状況に応じて変更になる場合があります。",
    "モデル学習の実験にかかる Google Colab Pro の料金は補助されます。",
    "学部・学年・バックグラウンドを問わず、どなたでも参加できます。",
    "サークルのモットーは「楽しい環境で一緒に AI を学ぼう」です。"
]

QUESTION = "KaiRA のモットーは何ですか?"


# ---------- 日本語 LLM (4-bit) ----------
DEV, BASE = "cuda", "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto",
                                            quantization_config=bnb_cfg)
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r               = 8,
    lora_alpha      = 16,
    lora_dropout    = 0.05,
    target_modules  = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # FFN
    ],
    bias      = "none",
    task_type = "CAUSAL_LM",
)
model = get_peft_model(copy.deepcopy(base), lora_cfg).eval()

@torch.no_grad()
def embed(text: str) -> torch.Tensor:
    ids = tok(text, truncation=True, max_length=128,
              return_tensors="pt").input_ids.to(DEV)
    vec = model.base_model.get_input_embeddings()(ids).mean(1)
    return (vec / vec.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

# ---------- bpb (bits-per-byte) 計算 ----------
def calc_bpb(texts):
    """
    texts: list[str]
    戻り値: Bits-per-byte (float)
    """
    loss_sum = 0.0
    byte_sum = 0
    for t in texts:
        enc = tok(t, return_tensors="pt").to(DEV)
        with torch.no_grad():
            loss_val = model(**enc, labels=enc.input_ids).loss.item()
        loss_sum += loss_val * enc.input_ids.size(1)
        byte_sum += len(t.encode("utf-8"))
    return loss_sum / byte_sum / math.log(2)   # nat → bit

# ---------- build Faiss index ----------
doc_emb = torch.stack([embed(t) for t in RAW_PARAS]).numpy().astype("float32")
index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(doc_emb)

# ---------- Pre-FT ----------
pre_bpb = calc_bpb([QUESTION])
answer_pre = tok.decode(
    model.generate(**tok(QUESTION, return_tensors="pt").to(DEV),
                   max_new_tokens=40, do_sample=False)[0],
    skip_special_tokens=True
)

print("===== 事前状態 (Pre-FT) =====")
print("質問:", QUESTION)
print("回答:", answer_pre.strip())
print(f"bpb  : {pre_bpb:.3f}")
print("=============================")

# ---------- sweep λ ----------
noise_floor = 1e-6
STEPS = 10
for lam in [0, 0.001, 0.01, 0.1, 1, 10, 100]:
    print(f"\nλ = {lam}")
    std = max(np.sqrt(lam), noise_floor)
    vtl = VTL(target=torch.empty(0), noise_std=std)
    retr = Retriever(index, acquisition_function=vtl,
                     device=torch.device(DEV))

    # K=None なら全件対象; (scores, idx, _, _)
    _, idx, _, _ = retr.search(embed(QUESTION)[None, :].numpy(), N=8, K=None)
    idx = idx.tolist()
    sel = [RAW_PARAS[i] for i in idx]

    print("選択された文書 ID:", idx, sel)
    print("-------- Fine-Tuning --------")
    ft_texts = sel + [f"### Question: {QUESTION}\n### Answer:"]
    batch = tok(ft_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(DEV)
    labels = batch.input_ids.clone()
    labels[labels == tok.pad_token_id] = -100

    opt = bnb.optim.Adam8bit(model.parameters(), lr=3e-4)
    model.train()
    for step in range(STEPS):
        loss = model(**batch, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        step_bpb = calc_bpb([QUESTION])
        print(f" step {step+1:02d}/{STEPS:02d} | "
              f"loss = {loss.item():.4f} | bpb = {step_bpb:.3f}")

    model.eval(); torch.cuda.empty_cache()

    # ---------- Post-FT ----------
    post_bpb = calc_bpb([QUESTION])
    answer_post = tok.decode(
        model.generate(**tok(QUESTION, return_tensors="pt").to(DEV),
                       max_new_tokens=40, do_sample=False)[0],
        skip_special_tokens=True
    )

    delta = pre_bpb - post_bpb
    print("-------- 結果 --------")
    print("回答:", answer_post.strip())
    print(f"bpb  : {post_bpb:.3f}  (Δ = {delta:+.3f}   ※マイナス=改善)")
    print("----------------------")
