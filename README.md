# TTFT-SIFT
論文実装のコンセプトを実験した記録とコード

京大のKaiRAのコード付き論文読み会で論文を発表します。
論文で実装したのと同じ環境は使えないのですが、T4 16GB 程度で動くサンプルコードを作成したので紹介します。
論文：https://arxiv.org/abs/2410.08020

論文と実験の環境やパラメータの差
論文で実行した環境と同程度のものを準備は難しかったため、T4でも動くように要点だけ試すコードを作成しました。
違いは以下の表にまとめてます。

| 項目 | 論文 (Hubotter et al.) | Vanilla_SIFT.py<br>*Vanilla SIFT* | Adaptive_SIFT.py<br>*Adaptive SIFT 疑似版* |
|------|-----------------------|---------------------------------|---------------------------------------------|
| **データ選択** | SIFT (候補 200 → 50) | SIFT (TOP 8) | SIFT (TOP 8) |
| **微調整ステップ数** | 1 step / doc × 50 doc ⇒ **50 step 固定** | **10 step 固定** | **最大 50 step だが σ 条件で可変** |
| **Early-Stopping ルール** | σₙ > (α n)⁻¹ | ― | bpb≈σ とみなし **α=0.25** で判定 |
| **Compute–Performance 比 α** | 0.15–0.5 で検証 | 固定コンピュート | 0.25 （変更可） |
| **評価指標** | bpb（Pile 全体） | bpb（質問 1 文）＋ loss | bpb（質問 1 文） |
| **σₙ の算出方法** | カーネル式で厳密計算 | 実装なし | bpb を直接 σₙ 近似 |
| **最大 step** | 50 | 10 | 50 |
| **実行時間 / VRAM** | RTX 4090：数秒 / prompt | 16 GB GPU：数十秒 | 早期停止で **さらに短縮** |
| **コード差分の要点** | ― | 基本形 (*Vanilla*) | - ALPHA, STEPS_MAX を追加<br>- ループ内で bpb 計算<br>- if σₙ > 1/(α·n) で break レポート |


### インストール (実験環境: Ubuntu 24.04, CUDA 12.6, cudnn 9.6, python3.11.11)
```
# pip install -r requirements.txt
```

### EARLY-STOPPINGのない固定10ステップで学習、様々なλで動かすコード
```
# python Vanilla_SIFT.py
```

### EARLY-STOPPINGを使い疑似的にAdaptive＿SIFTを模倣したコード
```
# python Adaptive_SIFT.py
```

またGOOGLECOLABのノートでも実験できます
現時点(2025年6月27日）では問題なく実行されております。

Vanilla_SIFT.ipynb
Adaptive_SIFT.ipynb
