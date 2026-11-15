# Research log (xmodel)

The working log of the project, kept as it was written: each section's design and pass/fail rule were written
before the run, and results were added after. Section numbers are the "README N" that the scripts' comments refer
to. Paths such as `cache/...` are local working folders that are not in this repository. Some early sections are in
Japanese. Personal details have been removed.

---

# xmodel：Qwen3.8-Flash-Next と Qwen3.8-27B の重みを比べる

Qwen3.8-Flash-Next（MoE、Qwen4 の試作アーキテクチャ）から 27B 分を切り出し、Qwen3.8-27B（密）との共通点・相違点を見たうえで、WeightFix の重みの書き換えで性能を上げられるか、という探索的な実験の第1段階です。
まずは Mac だけで、モデル全体をダウンロードせずに比べられる部分を比べました。

## 前提：2つのモデルの同じところ・違うところ（`config.json` とファイルのハッシュ）

- 完全に同じ：トークナイザー（`tokenizer.json`・`vocab.json`・`merges.txt`）と `chat_template.jinja` はファイルのハッシュまで一致。語彙 248,320。
- 同じ形：Gated DeltaNet のヘッド構成（キー 16・値 48、各 128 次元、畳み込みカーネル 4）、attention のヘッド 24・head_dim 256、RoPE の設定、4層に1層が attention。
- 違う：hidden size（2,560 と 5,120）、層の数（48 と 64）、KV ヘッド（2 と 4）、MLP（MoE 512 個と密 17,408）、Flash-Next だけにある GR・QSA・PLE、GDN の出力ゲート（sigmoid と swish）。

## 仕組み

- `remote_st.py`：safetensors のヘッダーと必要なバイト範囲だけを HTTP の範囲指定で取る。シャードを丸ごと落とさない。
- 取ったテンソルは `xmodel/cache/` に保存する（git の管理外）。

## 1. GDN のヘッドごとのパラメータ（`gdn_compare.py` → `results/xmodel_gdn.json`）

形が同じ `A_log`・`dt_bias`（48 ヘッド）、`norm.weight`（128）、`conv1d.weight`（10,240×4）を、linear attention の全層（Flash-Next 36 層、27B 48 層）で比べた。
片方の層ごとに、もう片方の全層の中で最も相関の高い層を探し、その相関の絶対値の平均を出した。偶然の水準は、要素の並びをシャッフルして同じ計算を 200 回行ったときの最大値の 95% 点。

| 比較 | A_log | dt_bias | norm | conv1d |
|---|---:|---:|---:|---:|
| Qwen3.5-27B ↔ Qwen3.8-27B（同じ系統の基準。同じ層どうし） | 1.000 | 1.000 | 0.997 | 0.996 |
| Flash-Next ↔ Qwen3.8-27B（最も合う層） | 0.345 | 0.304 | 0.208 | 0.021 |
| 偶然の水準（シャッフル、95% 点） | 0.544 | 0.538 | 0.386 | ― |

- Qwen3.8-27B は Qwen3.5-27B の続きから学習されている（GDN のパラメータがほぼ同じ値）。
- Flash-Next と 27B は、最も合う層を選んでも偶然の水準に届かない。conv1d はチャンネルの位置をそろえても相関 0.02。**Flash-Next は 27B から作られたものではなく、別々に学習されたと見てよい。**
- 言えないこと：ヘッドの並べ替えをしたうえで派生させた場合は、この比べ方では見分けられない（ただし conv1d の 10,240 チャンネルがまったく合わないので、可能性は低い）。

## 2. 埋め込みの形（`embed_compare.py` → `results/xmodel_embed.json`）

hidden size が違うので、値ではなく形を比べた。語彙の先頭 32,768 行（最初の 256 行のバイトのトークンは除く）を使う。
比較の基準として、同じトークナイザーを使う別のモデル Qwen3.5-35B-A3B（MoE、hidden 2,048）も並べた。

- RSA：4,000 トークンのトークン同士のコサイン類似度の行列が、どれだけ相関するか。
- kNN：各トークンの近い 10 トークンが、2つのモデルでいくつ一致するか。
- 線形写像：8 割のトークンで片方からもう片方への線形写像（リッジ回帰）を当てはめ、残り 2 割で確かめる。top1 は、写した先で一番近いのが正しいトークンだった割合（4,000 トークン中）。

| 埋め込み | 組 | RSA | kNN | 写像 top1 |
|---|---|---:|---:|---:|
| 入力（embed） | Flash-Next ↔ 27B | 0.485 | 0.441 | 97.6% |
| | Flash-Next ↔ 35B-A3B | 0.608 | 0.560 | 99.0% |
| | 27B ↔ 35B-A3B | 0.475 | 0.452 | 99.3% |
| 出力（lm_head） | Flash-Next ↔ 27B | 0.745 | 0.689 | 98.8% |
| | Flash-Next ↔ 35B-A3B | 0.775 | 0.685 | 99.9% |
| | 27B ↔ 35B-A3B | 0.744 | 0.692 | 99.9% |

シャッフルした対応では、RSA 0.00、kNN 0.002〜0.003、top1 0.00〜0.08%。

- どの組も、形はかなり似ていて、**線形写像でほぼ 1 対 1 に移せる**（top1 97.6〜99.9%）。同じトークナイザーで学習したモデルどうしなら、この程度は似る。
- Flash-Next と 27B が特に近いわけではない。27B ↔ 35B-A3B と同じ水準で、入力の埋め込みでは Flash-Next は 35B-A3B の方に近い（MoE どうし、または hidden size が近いせいかもしれない。区別はしていない）。
- 写像の R² は Flash-Next → 27B で 0.09 と低い（35B-A3B との組は 0.23〜0.63）。小さい次元から大きい次元へ写すと、説明できない成分が多く残る。トークンを見分ける情報は移せるが、ベクトル全体は移せない。

## ここまでの結論と次へのつながり

- **重みとしての「共有部分」はない。** GDN のパラメータに相関はなく、別々に学習されている。共通しているのは、トークナイザーと、表現の形（線形写像で移せる程度に似ている）だけ。
- なので、次の段階で Flash-Next と 27B をつなぐ物差しは、重みの比較ではなく**出力の比較**にする。トークナイザーが同じなので、同じ文章に対する次のトークンの確率を、そのまま KL で比べられる。
- 計画：Flash-Next から切り出したもの（またはその練習台）のルーターのつまみを、27B の出力に近づくように書き換え、WeightFix と同じくランダム・ES・勾配と比べる。

# 練習台：Qwen3.5-35B-A3B を切り出して、ルーターの書き換えで取り戻せるか

ここは結果を見る前に固定する（2026-11-07）。変えるときは末尾の変更履歴に理由と日時を書く。

## 問い

エキスパートを切り出して小さくした MoE で、**ルーターのつまみ（層ごと・エキスパートごとの1つの数）だけ**を書き換えると、切り出しで失った分をどれだけ取り戻せるか。
取り戻せる余地（伸びしろ）が勾配でも小さいなら、LLM に書き換えさせる段階（WeightFix の本題）に進む意味はない。まずそれを確かめる。

## なぜ 35B-A3B か

Flash-Next と同じ系統のルーター（softmax → 上位 k 個 → 正規化、バイアスなし）で、Colab に載る大きさにできる。
hidden 2,048、40 層、エキスパート 256 個（上位 8 個）、1 個 315 万パラメータ、エキスパートの合計 32.2B、エキスパート以外 約 3.8B。

## 切り出し方

- Flash-Next → 27B と同じ割合を残す：**1 層あたり 48 個（18.75%、全体 約 9.8B）**。参考に 104 個（40.6%、約 16.9B）。
- 全層で同じ個数（HF の実装が全層同数を前提にしているため）。
- 残すエキスパート：較正データでの「割り当てられた重みの合計」（正規化後のルーターの重みを、トークンについて足したもの）が大きい順。
- ルーターは消したエキスパートの行を除くだけ。残った中で softmax・上位 8 個・正規化が自動でやり直される。

## つまみ

各層・残した各エキスパートに 1 つ、ルーターの logit に足すバイアス b（48 個なら 40 × 48 = 1,920 個）。b = 0 が書き換えなし。
選ばれる確率と重みの両方に効く。HF のルーターにはバイアスがないので、実験ではルーターの計算を差し替えて足す（そのまま配るには推論側の対応が要る。練習台では気にしない）。

## データと物差し

- 文章：日本語と英語の Wikipedia（記事単位で分ける）。512 トークンの区切りで、較正 C 128 本、選択用 A 32 本、報告用 B 32 本。C・A・B は別の記事。
- 先生：切り出す前の 35B-A3B。A と B での次のトークンの確率の上位 64 個（と残りの合計）を保存する。
- 主要指標：B での KL（先生 → 切り出したもの）。`closed = 1 − KL(書き換え後) / KL(書き換えなし)`。切り出す前は KL = 0 なので、失った分のうちどれだけ取り戻したかになる。
- 信頼区間：B の 32 本を単位とするブートストラップ（95%）。
- 副指標：B の正解トークンでの交差エントロピー（パープレキシティ）。参考：Qwen3.8-27B を先生にした KL（別に計算できた場合だけ）。

## 比べる手法（予算は A の評価 64 回分）

| 手法 | 内容 |
|---|---|
| none | 書き換えなし（b = 0） |
| heuristic | 消したエキスパートが受け持っていた重みを、ルーターの行が最も似ている残したエキスパートに足したものを目標にし、C で測った今の重みとの比の対数で b を決める（3 回くり返す）。読んで決める規則の代わり |
| random | ランダムに選んだ数か所の b をずらし、良くなったら残す |
| es | 対角のガウス ES（対になる2点で評価） |
| grad | KL を下げるように Adam で b を更新（パラメータは b だけ） |

A の小分け（8 本）で評価したときは 1/4 回と数える。選択は A、報告は B。

## 判定基準

- **H1（伸びしろ）**：48 個の切り出しで、grad の `closed` が 0.10 以上、かつ信頼区間の下限が 0 より大きい。
- **H2（勾配なしで届くか）**：heuristic・random・es のうち最良の `closed` が、grad の半分以上。
- H1 を満たさなければ「ルーターのバイアスだけでは、この切り出しの損失をほとんど取り戻せない」と結論し、LLM に書き換えさせる段階には進まない。次の候補はエキスパートの統合や LoRA。
- H1 を満たし H2 を満たさなければ、伸びしろはあるが勾配が要る問題で、LLM の書き換えは勾配と比べて不利な見込み（WeightFix v1 の学習不足のネットと同じ形）。
- 両方満たせば、Flash-Next での本番と、LLM に書き換えさせる段階を設計する。
- 104 個の結果は参考で、判定には使わない。

## 結果（2026-11-08、48 個、先生は元の 35B-A3B）

生の出力は `results/xmodel_q35a3b/`（`edit_48_teacher.json`、各手法のバイアス、`router_stats.npz`、先生の出力 `teacher.npz`、データ `data.npz`）。

**切り出しの損失**：残した 48 個（256 個中）が C で受け持っていた重みは、層の平均で 59.5%（最も少ないのは第 2 層の 38.5%、最大 68.9%）。104 個では平均 79.9%。48 個に切ると、B の交差エントロピーは 2.08 → 2.69（+0.60）、先生との KL は 0.611。

| 手法 | 予算 | KL（A、選択用） | KL（B、報告用） | CE（B） | `closed`（B）と 95% 信頼区間 | 平均 \|b\| | 秒 |
|---|---|---|---|---|---|---|---|
| none | 0 | 0.723 | 0.611 | 2.687 | — | 0 | 16 |
| heuristic | 12 | 0.899 | 0.795 | 2.884 | −0.30 [−0.36, −0.25] | 0.58 | 26 |
| random | 64 | 0.720 | 0.616 | 2.693 | −0.01 [−0.01, −0.00] | 0.02 | 118 |
| es | 62 | 1.030 | 0.950 | 3.036 | −0.55 [−0.66, −0.46] | 0.66 | 115 |
| grad | 63 | 0.637 | 0.648 | 2.720 | −0.06 [−0.09, −0.04] | 0.49 | 571 |

**判定：H1 は満たさない**（grad の `closed` は B で −0.06）。H2 は grad がマイナスなので判定できない。事前に決めた通り、「ルーターのバイアスだけでは、この切り出しの損失をほとんど取り戻せない」と結論し、Flash-Next での本番と LLM に書き換えさせる段階には、この方法のままでは進まない。

読み取れること：

- **勾配でも取り戻せる量が小さい。** grad は A（学習に使った 32 本）では KL を 12% 下げたが、B では 6% 悪化した。つまみ 1,920 個に対して 32 本で約 500 歩回したので過学習している。ただし学習に使った A でさえ 12% しか縮まらないので、損失の大部分は「選び方のずれ」ではなく、消したエキスパートの中身そのものだと考えられる。
- **ルーターの重みの配分を元に合わせても、出力は元に近づかない。** heuristic はむしろ悪化した。消したエキスパートの分を最も似た残りに回すと、そのエキスパートが本来向いていないトークンまで受け持つことになる。
- **es の結果は手法の設計ミスで、勾配なしの手法全般についての証拠ではない。** 1,920 次元に対して 1 歩で試す方向が 4 組しかなく、勾配の推定はほぼノイズになる。正規化した歩幅 0.3 で、ほぼ当てずっぽうに歩いた（平均 |b| 0.66）。
- A と B では none の KL が 0.72 と 0.61 と違う。それぞれ 32 本しかないので、ばらつきは大きい。

次の候補（どれも未着手）：

1. **エキスパートの統合**：消すエキスパートの重みを、割り当て先の残すエキスパートに、受け持っていた重みの比で混ぜる。中身を戻せる唯一の「重みだけ」の方法。
2. **残すエキスパートの LoRA**：先生の KL で少し学習する。勾配は使うが、取り戻せる上限を測れる。
3. **残す数を増やす**：104 個（重み 80%）での損失を測っていない。今回は時間を優先して回していない。

# 練習台その2：消すエキスパートを残すエキスパートに統合する（2026-11-08、事前に決めた設計）

## 問い

ルーターのバイアスでは取り戻せなかった（損失の大部分は消したエキスパートの中身）。では、消すエキスパートの中身を、残すエキスパートの重みに直接書き込めば取り戻せるか。形は切り出しと同じ（48 個、中間 512）なので、そのまま普通のチェックポイントとして使える。

## 統合の仕方（`merge_pass.py`）

どの変種も、残す 48 個とそれぞれの割り当て（消すエキスパートごとに、ルーターの行が最も似ている残すエキスパート）は第1段階と同じ（`router_stats.npz`）。ルーターの行は変えない。

| 変種 | 内容 |
|---|---|
| none | 切り出しのみ（第2段階の none と同じはず。流しながらの計算と、保存したモデルでの計算が一致するかの確認も兼ねる） |
| avg | 割り当てられたエキスパートの重み（gate・up・down）を、C で受け持っていた重みの比で平均する |
| align | avg の前に、中間ニューロンの並びをそろえる。ニューロンごとに (gate の行, up の行, down の列) をつないだベクトルのコサイン類似度で、残すエキスパートのニューロンとの対応をハンガリー法で決める |
| lsq | gate・up は切り出しのまま、down だけを最小二乗で書き直す。C のトークンで、元のモデルのエキスパート部分（共有エキスパートを除く）の出力に、切り出したモデルの出力が近づくように、48 個の down を一度に解く（中間 512 × 48 = 24,576 次元のリッジ回帰、今の down に向かって縮める）。リッジの強さは A でその層の誤差が最小になるものを選ぶ（「書き直さない」も候補に入れる） |
| align+lsq | align の上で lsq |

- 勾配は使わない。lsq は閉じた形で解ける線形の書き換え。
- 層ごとに順に作る。各変種は自分の前の層までの出力を入力にする（元のモデルの入力ではない）。目標は「その入力に元の層をかけた出力」。
- 流しながら A・B の最終出力まで計算し、先生（`teacher.npz`）との KL を出す。全変種のチェックポイントも保存する（後でベンチマークに使う）。

## 判定基準

- **H3（統合で取り戻せるか）**：avg・align・lsq・align+lsq のうち A の KL が最小のものが、B で `closed` 0.10 以上、かつ信頼区間の下限が 0 より大きい。
- H3 を満たせば：そのモデルを 3 分野のベンチマーク（理科・数学・コード）で、切り出しのみと比べる。KL だけでなく実際の性能でも差が出れば、Flash-Next での本番を設計する。
- H3 を満たさなければ：LoRA で取り戻せる上限を測るか、残す数を増やす。
- 参考：none の KL が第2段階の 0.611 と大きくずれたら、流しながらの計算か保存に問題があるので、結果は使わない。

## 結果（2026-11-08、48 個、先生は元の 35B-A3B）

生の出力は `results/xmodel_q35a3b/merge_48_teacher.json`（層ごとの lsq の診断つき）、配列ごとの KL は `merge_48_teacher_kl.npz`、ログは `merge_48.log`。A100 40GB で 61 分（1 層あたり 73〜101 秒）、準備込みで約 8 CU。

| 変種 | KL（A） | KL（B） | CE（B） | `closed`（B）と 95% 信頼区間 |
|---|---|---|---|---|
| none | 0.723 | 0.611 | 2.687 | — |
| avg | 1.811 | 1.837 | 3.954 | −2.01 [−2.54, −1.59] |
| align | 1.630 | 1.635 | 3.742 | −1.68 [−2.13, −1.32] |
| lsq | 0.717 | 0.608 | 2.681 | +0.005 [−0.013, +0.025] |
| align+lsq | 1.140 | 1.052 | 3.127 | −0.72 [−0.86, −0.62] |

none は第2段階（保存した切り出しモデルで 0.611）と一致したので、流しながらの計算は信用できる。

**判定：H3 は満たさない。** A で選ばれたのは lsq で、B の `closed` は +0.005（信頼区間が 0 をまたぐ）。事前に決めた通り、統合はベンチマークに進めない。

読み取れること：

- **重みの平均は、ただ消すよりはるかに悪い。** avg は KL を 3 倍にした。ニューロンの並びをそろえても（align）ほとんど変わらない。同じ層のエキスパートどうしでも、重み空間で混ぜられるほど似ていない。しかも残す側は受け持ちの大きいエキスパートなので、混ぜるとその本来の仕事が壊れる。
- **down の書き直しは、層の中では効くのに、最後の出力にはほとんど効かない。** lsq は層ごとにその層の誤差を平均 17%（最大 42%）減らしたが、最終の KL は 0.5% しか縮まらなかった。align+lsq は層の中で平均 33% 減らしたが、出発点（align）が悪すぎて none に届かない。gate・up を固定したままの線形の書き直しでは、消えたエキスパートが作っていた特徴を作れない。
- ルーターのバイアス（練習台その1）と合わせると、48/256 個（受け持ち 60%）への切り出しでは、**重みだけの書き換え（勾配なし）で取り戻せる損失はほぼない**。損失は消したエキスパートの中身そのもので、取り戻すには学習（蒸留・LoRA）か、残す数を増やすしかない。
- Flash-Next から 27B 分（約 15%）を切り出す本番にとっては悪い知らせ。残す割合はこの練習台（19%）より小さい。

**次の候補（どれも未着手）：**

1. **LoRA で上限を測る**：残した 48 個（または down だけ）を、先生の KL で少し学習する。勾配を使えば取り戻せるのかどうかが分かる。
2. **残す数と損失の関係**：104 個（受け持ち 80%）、さらに 160 個などで none の KL を測り、どこから損失が急に増えるかを見る。第1段階の流しを 1 回やれば、いくつでも同時に測れる。
3. lsq を gate・up まで広げる（非線形なので、閉じた形では解けない。交互に解くか勾配が要る）。

# 練習台その3：LoRA で学習させるとどこまで戻るか（2026-11-08、事前に決めた設計）

## 問い

重みだけの書き換え（ルーターのバイアス、統合、down の最小二乗）では、48 個への切り出しの損失はほぼ戻らなかった。勾配で学習させれば戻るのか。戻るなら、どのくらいのデータで、どこに LoRA を付ければよいか。

## やり方

- **学習データ T**：Wikipedia の日英 2,048 本（512 トークン、1 記事 1 本、半分ずつ）。C・A・B と同じ記事は除く。先生（元の 35B-A3B）の上位 64 個の対数確率を、第1段階と同じ流しで T にも付ける（`prune_pass.py --extra-train 2048`）。この流しで 48 個の切り出しも作り直す。C は同じなので、残すエキスパートは前回と同じになる。
- **損失**：先生との KL（上位 64 個と残り 1 つ、これまでの物差しと同じ）。
- **変種**（どちらも切り出しのみのモデルから始める。ルーターは学習しない）：
  - `lin`：デコーダー層の全ての線形層（注意、GDN、共有エキスパート）に LoRA（r 16、α 32）。
  - `lin+exp`：`lin` に加え、残した 48 個のエキスパートの gate_up と down にも、エキスパートごとの LoRA（r 8、α 16）。
- 学習率 2e-4（最初の 20 歩で上げ、コサインで 0 まで下げる）、AdamW、1 歩 4 本、T を 1 周（512 歩）、勾配のノルムは 1 で切る。64 歩ごとに A で測り、A が最小のときの LoRA を使う。報告は B。

## 判定基準

- **H4**：`lin+exp` の B の `closed` が 0.25 以上、かつ信頼区間の下限が 0 より大きければ「学習すれば戻る」。0.10 未満なら「このデータ量では学習しても戻らない」。間は「一部戻る」。
- `lin` と `lin+exp` の差で、損失が「エキスパートの中身」にあるのか、「周りの層が合わせれば済む」ものなのかを見る。
- 参考：学習前（0 歩目）の KL が 0.611 と大きくずれたら、読み込みに問題があるので結果は使わない。

## 1 回目の実行（2026-11-08、途中で VM を失った）

- 教師の流し（T 2,048 本込み）は 61 分で終わり、T の教師 CE は 1.896。
- `lin_exp`（学習するパラメータ 1.08 億）：0 歩目の KL は A 0.7224・B 0.6116（第2段階と一致）。64 歩（T の 1/8、13 万トークン）で A の KL 0.531（A で 27% 縮んだ）。1 歩 10.8 秒で、512 歩なら約 92 分の見込みだった。
- その約 20 分後に VM が回収され、学習の途中経過も教師の T も切り出しも消えた。原因は、ログの確認を `colab exec` から `colab download` に切り替え、カーネルが 20 分ほど何も実行しなかったことだとみている（`colab log` で、最後の exec から約 21 分後に課金が止まっている）。`colab_bg.sh tail` に、毎回カーネルを叩く処理を足した。
- 判定はまだできない（B は測っていない）。

## 結果（2026-11-08、2 回目の実行、出力は Google Drive）

生の出力は `results/xmodel_q35a3b/lora_48_lin_exp.json`・`lora_48_lin.json`（学習曲線つき）とログ `job.log`。LoRA の重み（`lora_48_*.pt`）、切り出し（`pruned_48`）、T の教師は Drive の `MyDrive/xmodel/q35a3b`（17GB）にある。教師の流し 44 分、`lin_exp` の学習 85 分、`lin` の学習 11 分。準備込みで約 20 CU。

| | 学習するパラメータ | KL（A） | KL（B） | CE（B） | `closed`（B）と 95% 信頼区間 |
|---|---|---|---|---|---|
| 学習前（切り出しのみ） | — | 0.722 | 0.611 | 2.687 | — |
| `lin`（線形層だけ） | 2,100 万 | 0.486 | 0.400 | 2.468 | 0.345 [0.319, 0.373] |
| `lin+exp`（＋エキスパート） | 1 億 800 万 | 0.465 | 0.381 | 2.449 | 0.376 [0.350, 0.407] |
| 参考：元のモデル | — | 0 | 0 | 2.084 | 1 |

学習曲線（A の KL、64 歩ごと）：`lin+exp` は 0.530 → 0.503 → 0.489 → 0.478 → 0.473 → 0.468 → 0.466 → 0.465。どちらも最後の歩が最良で、学習データと A の KL の差は小さい（合わせすぎは目立たない）。

**判定：H4 は満たす（どちらの変種も「学習すれば戻る」）。** 重みだけの書き換え（`closed` ほぼ 0）と違い、T の 1/8（13 万トークン）でもう 27% 戻る。CE で見ると、切り出しで増えた 0.60 のうち 0.24（39%）が戻った。

読み取れること：

- **戻る分の大半は、エキスパートの外で取り戻せる。** 線形層（注意・GDN・共有エキスパート）だけの `lin` で 34.5%、エキスパートにも付けた `lin+exp` で 37.6%。差は 3 ポイントで、パラメータは 5 倍、学習時間は 8 倍（エキスパートごとの LoRA を Python のループで回しているため）。消したエキスパートの中身を、残したエキスパートの中に作り直す必要はあまりなく、周りの層が合わせれば済む部分が大きい。
- **1 周（約 100 万トークン）では頭打ちに近い。** 最初の 64 歩で 27%、残りの 448 歩で 11 ポイント。残り 6 割強の損失を取り戻すには、データ量・周回・ランクのどれかを増やす必要がある（どれが効くかは未確認）。
- 「重みを直接書き換える」発想（WeightFix）からすると、勾配なしではほぼ 0、勾配ありの LoRA で 35% 前後、という差がはっきり出た。勾配を使わずにこの差を埋める方法は、今のところ手がかりがない。

**次の候補：**

1. 事前に決めた通り、3 分野のベンチマーク（理科・数学・コード）で、元のモデル・切り出しのみ・切り出し＋LoRA を比べる。LoRA は本体の重みに足し込めば普通のチェックポイントになり、vLLM でそのまま動く。元のモデルは A100 40GB に載らないので、公式の FP8 版を使うか、80GB の GPU にする。
2. `lin` は安い（11 分）ので、データ量やランクを変えて、どこまで戻るかの曲線を取る。

## 切り分け：ランクか、データ量か（2026-11-08、事前に決めた設計）

`lin+exp` では、学習データの KL も 256 歩目から 0.41〜0.43 で止まり、A との差も小さかった（学習データにすら合わせきれていない）。容量（ランク）が足りないのか、データが足りないのかを、安い `lin` で切り分ける。基準は `lin`（r 16、学習率 2e-4、T 2,048 本 1 周、`closed`（B）0.345）。

| 実験 | 変えるもの | 読み方 |
|---|---|---|
| `r64` | ランク 64 | 基準より信頼区間が重ならないほど上がれば、ランクが効く |
| `r256` | ランク 256 | r64 からさらに上がるか、頭打ちか |
| `lr1e-3` | 学習率 1e-3 | 上がれば、そもそも学習率が低すぎた |
| `half2ep` | T の前半 1,024 本を 2 周（歩数は同じ） | 基準と差がなければ、データ量は効いていない（増やしても伸びない見込み）。下がれば、データを増やす実験（教師の作り直し 44 分）に進む |

### 結果（2026-11-08）

生の出力は `results/xmodel_q35a3b/lora_48_lin_<tag>.json` とログ `ablation.log`。4 本で約 50 分、準備込みで約 9 CU。

| 実験 | 学習するパラメータ | KL（A） | KL（B） | CE（B） | `closed`（B）と 95% 信頼区間 | 学習データの KL（最後の 64 歩） |
|---|---|---|---|---|---|---|
| 基準（r 16） | 2,100 万 | 0.486 | 0.400 | 2.468 | 0.345 [0.319, 0.373] | 0.459 |
| `r64` | 8,500 万 | 0.472 | 0.387 | 2.455 | 0.367 [0.340, 0.397] | 0.442 |
| `r256` | 3 億 3,900 万 | 0.473 | 0.394 | 2.463 | 0.355 [0.326, 0.386] | 0.443 |
| `lr1e-3` | 2,100 万 | 0.487 | 0.406 | 2.473 | 0.336 [0.307, 0.366] | 0.455 |
| `half2ep` | 2,100 万 | 0.491 | 0.407 | 2.476 | 0.334 [0.307, 0.363] | 0.345 |
| 参考：`lin+exp` | 1 億 800 万 | 0.465 | 0.381 | 2.449 | 0.376 [0.350, 0.407] | 0.433 |

**どれも基準と信頼区間が重なる。ランクも、学習率も、この範囲のデータ量も、ほとんど効かない。**

- **ランク**：16 → 64 で +2 ポイント、256 ではかえって +1 ポイント。容量を 16 倍にしても伸びない。
- **学習率**：5 倍にしても変わらない（わずかに下）。学習率が低すぎたわけではない。
- **データ量**：前半 1,024 本を 2 周すると、学習データの KL は 0.345 まで下がる（基準は 0.459）のに、A・B はわずかに悪い（−1 ポイント）。つまり、モデルには学習データに合わせる余力はある（同じ文章を 2 度見れば覚える）。それでも初めて見る文章には効かない。異なる文章を倍（1,024 → 2,048 本）にして得たのは +1 ポイントだけ。
- 学習前は「学習データにすら合わせきれていないので、容量が足りない」とみていたが、これは外れ。1 周目の学習データの KL は、「初めて見る文章」での値なので、A と同じように頭打ちになっていただけだった。

読み取れること（この時点）：LoRA（1 周、約 100 万トークン）で戻るのは 35% 前後で、これは設定を変えても動きにくい天井に見える。

## 物差し合わせ・ルーター・エキスパート側のランク・ベンチマーク（2026-11-08、G4）

A100 が空かず H100 は使えなかったので、G4（RTX PRO 6000 Blackwell、96GB、10.05 CU/時）で行った。学習は A100 の約 2 倍速い（`lin` が 640 秒 → 302 秒）。

### 物差し合わせ（`ref_kl.py`、同じ A・B の教師）

| モデル | パラメータ | KL（B） | CE（B） |
|---|---|---|---|
| Qwen3.5-9B（密） | 9.4B | 0.255 | 2.244 |
| Qwen3.5-4B（密） | 4.5B | 0.360 | 2.414 |
| 切り出し＋`lin+exp` | 8.47B（アクティブ 3.44B） | 0.381 | 2.449 |
| 切り出しのみ | 8.47B（アクティブ 3.44B） | 0.611 | 2.687 |

切り出し＋LoRA は、Wikipedia の KL ではきちんと学習した 4B にわずかに届かない。9B を目標の線とすると、切り出しのみから 9B までの差の約 65% を埋めた。

切り出しのパラメータ：エキスパート 6.04B（48 個 × 40 層）、埋め込みと出力層 1.02B、その他 1.41B、計 8.47B。1 語あたりに使うのは 8 個のエキスパートで、アクティブは元と同じ 3.44B（入力の埋め込みを除くと 2.93B）。

### ルーターとエキスパート側の容量（B の `closed`）

| 実験 | 学習するパラメータ | KL（B） | `closed`（B） |
|---|---|---|---|
| `lin`（基準） | 2,100 万 | 0.400 | 0.345 |
| `lin` ＋ルーター（`router`） | 2,500 万 | 0.400 | 0.345 [0.319, 0.373] |
| `lin+exp`、エキスパート r 8（新しい速い実装、`exp8fast`） | 1 億 800 万 | 0.381 | 0.376 [0.348, 0.407] |
| `lin+exp`、エキスパート r 64（`exp64`） | 7 億 1,300 万 | 0.411 | 0.327 [0.296, 0.361] |

- **ルーターは効かない**（学習率 2e-4 の float32 の差分、ゼロから）。
- **エキスパート側のランクを上げると悪くなる**。r 8 → 64 で 0.376 → 0.327。学習の初めから遅い（64 歩目の A の KL 0.581、r 8 は 0.530）。原因は未確認（LoRA の倍率 α = 2r のまま容量を増やしたので、実質の学習率が大きすぎた可能性）。
- エキスパートの LoRA を速くした実装（専門家ごとの同期をなくし、LoRA の計算を活性の型で行う）は、float32 で古い実装と一致し（損失は同じ、勾配の差 7e-7）、bf16 の学習でも同じ結果になった（0.3755 と 0.376）。1 歩 10.8 秒（A100、古い実装）→ 2 秒（G4、新しい実装）。

### ベンチマーク（Thinking あり、G4、vLLM 0.33）

設定：Thinking あり、最大 16,000 トークン、温度 0.6・top-p 0.95・top-k 20（問題ごとに種を固定）。予定は 5 モデル × 各分野 100 問だったが、元のモデルで 100 問に約 1,000 秒かかり、CU が足りないので途中で変えた（`bench_rest.sh`）：元のモデルは 300 問、9B・4B は各分野 50 問、切り出しの 2 つは各分野 20 問（どれも同じ順番の先頭なので、小さい方は大きい方に含まれる）。vLLM は `--max-num-seqs 64` が要る（既定の 1,024 では、GDN の状態の置き場（610）が足りずに起動しない）。

**切り出したモデルは、チャットモデルとして壊れている。** 確かめの 9 問（各分野 3 問）は全部不正解で、答えは質問文の繰り返しか、`<think>` の繰り返しだけだった。同じ仕組みで元のモデルはきちんと考えて答えるので、仕組みの問題ではない。

結果（`results/xmodel_q35a3b/bench_think/`、`grades_all.txt`）：

| モデル | 問題数 | 理科 | 数学 | コード | 上限 16k で打ち切り | 平均の生成トークン |
|---|---|---|---|---|---|---|
| 元の 35B-A3B | 各 100 | 87% | 77% | 93% | 24 / 300 | 6,052 |
| Qwen3.5-9B | 各 50 | 94% | 56% | 86% | 22 / 150（うち数学 21） | 5,629 |
| Qwen3.5-4B | 各 50 | 80% | 66% | 84% | 30 / 150 | 7,441 |
| 切り出しのみ | 各 20 | 0% | 0% | 0% | 48 / 60 | 12,972 |
| 切り出し＋`lin+exp`（KL 37.6% 回復） | 各 20 | 0% | 0% | 0% | 55 / 60 | 14,686 |

- 元のモデルとの差（同じ問題）：9B は理科 +8（[0, +18]）・数学 −24（[−36, −12]）・コード −8（[−16, −2]）。4B は理科 −6・数学 −14・コード −10。
- **9B の数学が 4B より低いのは、16k の上限のせい**。9B は数学 50 問中 21 問で、答える前に打ち切られた。上限を 32k にすれば変わるはず（未確認）。
- **LoRA で Wikipedia の KL は 37.6% 戻ったが、生成は 0% のまま。** `</think>` で考え終わることがほとんどなく（60 問中 2 問）、\boxed{} も使わない。`</think>` と \boxed{} を求めない甘い採点にしても、理科 3/20（10 択の偶然の水準）、数学 0/20。中身が通る答えもまれにある（「So, I should choose A.」）が、正答率は上がらない。
- KL（Wikipedia）と性能の関係：健全なモデル（元・9B・4B）の間では KL の順に性能も並ぶが、切り出しの系統は KL がそこそこでも生成は全滅。Wikipedia の KL は、この壊れ方をまったく見ていない。
- G4 で約 1 時間 45 分、約 18 CU。残りは 14.12 CU。

原因の見立て：**残すエキスパートの選択（C）も、LoRA の学習（T）も、KL の物差し（A・B）も、すべて Wikipedia だった。** チャットの書式・コード・数式を受け持つエキスパートが Wikipedia での受け持ちが少ないために消され、KL はその損失を見ていない。「KL が低い＝性能が高い」とは限らない、の極端な例。次は、選択と学習にチャット・コード・数学の文章を混ぜる。

# 練習台その4：壊れたのは「残すエキスパートの選び方」として直す（2026-11-08、計画。11/23 の CU 追加後）

WeightFix 本体の結果を踏まえる。LLM の編集が効いたのは壊れ方が局所的なとき（v1、外れ値のセル：`closed` 0.926）で、外れ値のない学習不足のネットでは LLM の編集はランダム並み（0.022）、効いたのは 3 段階の貪欲探索（0.686）だった。今回の崩れは重みの値ではなく、消したエキスパートの情報がないことなので、8.47B の重みを直接直すのは筋が悪い（ルーター編集・統合はほぼ 0%）。直す対象を **「どのエキスパートを残すか」** に変える。

| WeightFix | ここ |
|---|---|
| 編集コマンド `set W1 行 列 値` | `swap 層 残す 消した`（残す 48 個のうち 1 つを入れ替える） |
| 種 A で採点し、最良を採用 | チャット形式の A で KL を測り、最良を採用 |
| 外れ値のセルを見つける | Wikipedia では受け持ちが少ないのに、チャット・コード・数学では多いエキスパートを見つける |
| 学習不足のネットでは貪欲探索が効いた | 層ごとの入れ替えを貪欲に探す |
| 良かった試行で LLM を学習（FBC） | 良かった入れ替えで LLM の提案を学習（後回し） |

手順（各段階で、前の段階より B の KL と生成の正答率が良くなったときだけ次へ）：

1. **データを作り直す。** C・A・B・T を、Wikipedia に加えて、チャット（チャットテンプレートつき、Thinking の応答を含む）・コード・数学の文章を混ぜて作る。応答部分は元のモデル自身の生成にする（教師と同じ分布）。物差しの A・B もチャット形式にする。
2. **単純な規則で選び直す。** 混ぜた C での受け持ちの多い順に 48 個を残す（第1段階と同じ流し 1 回、約 45 分）。WeightFix v0 では固定ルール（0.630）が学習後の LLM（0.487）を上回ったので、まずこれが基準。
3. **入れ替えの探索。** 2 で戻りきらなければ、層ごとに `swap` の候補（混ぜた C での受け持ちの上位から）を A の KL で貪欲に選ぶ。比較として、層ごとの受け持ちの表（Wikipedia とチャットで分けたもの）を LLM に見せて `swap` を提案させる版も試す。
4. **LoRA をやり直す。** 混ぜた T で `lin+exp`（r 8）。
5. **生成で確かめる。** Thinking ありのベンチマーク（同じ設定）で、元のモデル・Qwen3.5-9B / 4B と比べる。

判定：2〜4 のどこかで、確かめの 9 問の挙動（質問文の繰り返し・`<think>` の反復）が消え、ベンチマークで 4B に近づけば「選び方の問題だった」。どの段階でも戻らなければ、48/256 個は少なすぎる（残す数を増やす）。残りの損失は、消したエキスパートが持っていた知識そのもので、取り戻すには知識を学び直すしかない。データを倍にして +1 ポイントなので、データを増やす方向はかなり効率が悪い（10 倍にしても数ポイントの見込み。未確認）。

## 分かっている弱点

- 先生の上位 64 個だけで KL を計算する（残りはまとめて 1 つとして扱う）。
- 較正と評価は Wikipedia だけ。会話・コード・数学での損失は測らない。
- バイアスは、切り出しの損失のうち「選び方のずれ」しか直せない。消えたエキスパートの中身そのものは戻らない。

## スクリプト（`xmodel/moe/`）

| ファイル | 内容 |
|---|---|
| `common.py` | 重みの読み出し（HF から 1 テンソルずつ、または手元のフォルダ）、1 層ずつの計算、データ、KL |
| `prune_pass.py` | 第1段階。元のモデルを 1 層ずつ流し、C でのルーターの選び方を数え、層ごとに切り出して保存し、元のモデルの出力（先生）を A・B で保存する。GPU に載るのは 1 層だけ |
| `router_edit.py` | 第2段階。切り出したモデルを読み、5 つの手法でバイアスを書き換え、B の KL・`closed`・判定を出す |
| `merge_pass.py` | 練習台その2。元のモデルを 1 層ずつ流し、5 つの変種（none・avg・align・lsq・align+lsq）を変種ごとの入力で作り、A・B の KL を出して全変種を保存する |
| `smoke_local.py` | 小さなランダムの MoE（4 層、エキスパート 16 個）で、第1・第2段階と統合を通す確認 |

### 手元での確認（2026-11-07、`smoke_local.py`）

- 1 層ずつの計算とモデル全体の計算で、交差エントロピーの差の最大は bf16 で 0.018、float32 で 1e-6 以下。bf16 の丸めの差だけで、計算の流れは同じ。
- 全エキスパートを残すと KL 0.0007（bf16 と先生の保存の丸め）。保存した切り出しのチェックポイントは `from_pretrained` で読める。
- 4 個に切ると KL 0.92 になり、5 手法とも予算内で動く。ランダムなモデルとランダムなトークンなので、手法の良し悪しはこの確認からは分からない。
- 統合（2026-11-08 追加）：全エキスパートを残すと 5 変種とも KL 6e-7（何も変わらない）。16 → 4 個では、流しながらの none（0.9164）と第1段階の切り出しチェックポイント（0.9170）が一致し、保存した align+lsq・avg のチェックポイントも流しながらの値と一致（0.7717 と 0.7722、0.8150 と 0.8148）。

## Colab での実行

A100 を 1 回のセッションで使う（切り出したチェックポイントは `/content` に置き、セッションをまたがない）。実測（2026-11-08、A100 40GB）は第1段階 46 分（約 71GB を HTTP で読む、1 層あたり約 70 秒）、第2段階（48 個、5 手法）は読み込み込みで約 15 分。準備と確認を含めて約 15 CU を使った。セッションを止めると切り出したチェックポイント（48 個で 16GB、104 個で 29GB）は消えるので、続きをやるときは第1段階からやり直す。

```bash
pip install -U "transformers>=5.17" datasets safetensors huggingface_hub
pip install -U flash-linear-attention   # 任意。GDN が速くなる
# xmodel/ を zip で上げて /content/xmodel に展開しておく
cd /content/xmodel/moe
python prune_pass.py --source Qwen/Qwen3.5-35B-A3B --out /content/runs/q35a3b --keep 48 104
python router_edit.py --run /content/runs/q35a3b --keep 48
python router_edit.py --run /content/runs/q35a3b --keep 104      # 参考
# 練習台その2（data.npz・teacher.npz・router_stats.npz があれば、第1段階の切り出しは要らない）
python merge_pass.py --source Qwen/Qwen3.5-35B-A3B --run /content/runs/q35a3b --keep 48
# 任意：Qwen3.8-27B を先生にする（約 54GB を読む）
python prune_pass.py --source Qwen/Qwen3.8-27B --out /content/runs/q35a3b --teacher-only --teacher-name teacher_27b
python router_edit.py --run /content/runs/q35a3b --keep 48 --teacher teacher_27b
```

### VM が回収されても続きからやり直せるようにする（2026-11-08 追加）

出力先（`--out` / `--run`）を Google Drive に置くと、VM が回収されても同じコマンドで続きから再開できる。

```bash
colab new -s xmodel --gpu A100
colab drivemount -s xmodel            # 初回はブラウザで Drive へのアクセスを許可する
RUN=/content/drive/MyDrive/xmodel/q35a3b
python prune_pass.py --source Qwen/Qwen3.5-35B-A3B --out $RUN --keep 48 --extra-train 2048 \
  && python lora_train.py --run $RUN --keep 48 --variant lin_exp \
  && python lora_train.py --run $RUN --keep 48 --variant lin
# VM が落ちたら、新しいセッションで drivemount して、同じ行をもう一度流すだけ
```

- `colab drivemount` の注意（2026-11-08 に分かったこと）：
  - 許可が済んだ合図（Enter）を `/dev/tty` から読むので、端末なしで裏で動かすと失敗する。`script -q <ログ> colab drivemount -s xmodel` のように疑似端末で動かす。
  - CLI は 10 分で待つのをやめる。許可がそれより遅れると「Credentials propagated」の後に時間切れになるが、認証は VM に届いている。カーネルを再起動してもう一度 `drivemount` すれば、許可を求められずにマウントできる。
  - 許可は VM ごとに要る見込み（許可画面は毎回 `prompt=consent`。新しいセッションで要るかは未確認）。
- `prune_pass.py`・`merge_pass.py` は 10 層ごと（`--ckpt-every`）に、隠れ状態と統計を `$RUN/_state_*.pt` に書く。もう一度流すと、そこから再開する（失うのは最大 10 層分、約 15 分）。それより前の層の保存ファイルがなければ止まる。
- `lora_train.py` は A で測るたび（64 歩ごと）に、LoRA・オプティマイザー・学習率の状態を書く。
- 出力がそろっている手順は「already done」と出して何もしない（やり直すときは `--force`）。だから一連の手順を `&&` でつないでおけば、落ちたところから先だけが動く。
- 状態は引数と一緒に保存し、引数が違えば使わない。終わったら消す。
- `smoke_local.py` の [7] で、途中で止めて再開した結果が、止めずに回した結果と一致することを確かめている。

長い処理は `xmodel/colab_bg.sh` で VM の中でバックグラウンド起動し、ログを短い問い合わせで確かめる（`colab exec` は終わるまで待ち、止まったり時間切れになったりするため）。

結果は `/content/runs/q35a3b/edit_<個数>_<先生>.json`（と、各手法のバイアス `_bias.npz`、ルーターの統計 `router_stats.npz`）。この 3 種類だけを持ち帰れば、ここで読み解ける。

# 性能が保たれているかのベンチマーク（`xmodel/bench/`）

KL は安い代わりの物差しで、本当に力が落ちていないかは問題を解かせて確かめる。理科・数学・プログラミングの 3 つ。

| 分野 | 問題 | 数 | 採点 |
|---|---|---:|---|
| 理科 | MMLU-Pro の物理・化学・生物（以前の評価と同じ 300 問・同じ指示文） | 300 | 最後の `\boxed{}` の記号 |
| 数学 | MATH-500 のレベル 4・5 | 262 | 最後の `\boxed{}` を math-verify で照合（無ければ正規化した文字列の一致。厳しめになる） |
| プログラミング | HumanEval+（EvalPlus の追加テスト付き） | 164 | 最後の ```python``` のコードにテストを付けて別プロセスで実行（30 秒で打ち切り） |

- `build_bench.py`：3 つを `tasks.jsonl` にまとめる（約 11MB、ほとんどが HumanEval+ のテスト。作り直せるので git には入れない）。各問に分野ごとの決まった順番 `order` を付け、問題数を絞るときも同じ問題を使う。
- `run_bench.py`：OpenAI 互換の API（vLLM、Strata など）に問い合わせる。既定は思考なし・上限 4,096・貪欲法（2 つのモデルを問題ごとに比べるときに、サンプリングの揺れを入れないため）。途中から再開できる。
- `grade_bench.py`：採点し、2 つ目以降の結果を 1 つ目と問題ごとに比べて、差とブートストラップの 95% 信頼区間を出す。コードの採点はモデルが書いたコードを実行するので、Colab の VM の中でだけ動かす。
- 差が見えなければ、プログラミングを LiveCodeBench の新しい問題に広げる。
- Flash-Next 本体は Strata（Niko1221/Strata、OpenAI 互換の API を立てられる）で Colab 上で動かせる見込み。生成は遅いので、本体の基準値は `--limit-per-task` で問題を絞って取る。

## Same budget, different shape (2026-11-09, Kaggle T4, `moe/width_pass.py`)

Every shape keeps 24,576 routed-expert neurons per layer (the 48×512 budget, ~8.5B total). Experts are picked by chat routed mass
and neurons by chat importance (chat calibration windows from an earlier run); no training. KL to the original on 60 held-out chat windows
(order 40–59). `full` (original layers) gives 0.0016, so the pass matches the teacher.

| shape | KL | chat mass kept | vs chat 48×512 (paired, 60 windows) |
|---|---:|---:|---|
| Qwen3.5-9B (dense, trained) | 0.119 | | 96×256 lower on only 5/60 |
| Qwen3.5-4B (dense, trained) | 0.143 | | 96×256 lower on only 6/60 |
| wiki 48×512 (the current cut) | 1.415 | 15% | |
| chat 48×512 (an earlier chat-only pick) | 0.325 | 47% | |
| 96×256 | **0.267** | 70% | KL −17.8% [−11.0, −23.5], better on 46/60 |
| 128×192 | 0.317 | 81% | −2.5% [+9.8, −13.1] (no difference) |
| 192×128 | 0.437 | 95% | +34% worse |
| 256×96 | 0.558 | 100% | worse |
| 192×128 top-16 | 0.525 | 95% | worse |

Choosing experts on chat instead of Wikipedia matters most (1.415 → 0.325). Spreading the budget over more, narrower experts helps a
little at 96×256, then hurts: neurons inside an expert are not redundant enough to cut by 75%. Chat 48×512 still loops when it
generates, so a 18% lower KL alone is unlikely to fix generation. Working models sit at 0.12–0.14 on this ruler
(9B/4B via `moe/ref_stream.py`, float32 on a Kaggle T4); the best untrained cut is still about 2x away. Results: `results/xmodel_q35a3b/width/`.

## 96×256 answering for real, and where the damage is (2026-11-09)

**Generation** (Mac, 4-bit MLX via `mac_shape.sh`; KL 0.288 vs 0.267 in float32; 60 questions, thinking on, 16k tokens;
code run on a disposable Kaggle VM; `loose` = the last answer anywhere, thinking included):

| model | sci (strict / loose) | math | code | total loose |
|---|---|---|---|---:|
| 96×256 4-bit | 0 / 0 | 0 / 0 | 1 / **6** | 6/60 |
| Qwen3.5-9B | 19 / 19 | 13 / 16 | 16 / 16 | 51/60 |
| earlier chat 48×512 | 0 | 0 | 2 (strict) | |

Only 2/60 answers close `</think>`; 58 run to 16k and end in loops. Starts are coherent and some code answers reach a
correct function inside the thinking (6/20 pass when that code is run), but science and math never state an answer:
the model calls ordinary words typos ("Pilocarpine is likely a typo"), then circles.

**Sensitivity** (`moe/sensitivity_pass.py`, Kaggle T4, KL on held-out chat windows; base 96×256 = 0.2587):

- Restoring the original MoE in any 5-layer group recovers 13–23% of the KL. The last 10 layers recover a bit more
  (22–23%). The damage is spread over all layers, with no single faulty spot for a local edit to fix.
- One try at a fix, spending the same budget unevenly (88–112 experts per layer, following the sensitivity), changed
  nothing: KL 0.2571, a drop of +0.6% [−0.6, +1.8].
- Probability of the end-of-thinking token where the original stops thinking: original 0.47, 96×256 0.19. Restoring
  layers 0–4 brings it to 0.31, more than any other group. The cut model does not stop, and loops follow.

Conclusion: at an 81% cut without training, the best shape (96×256, chat-picked) keeps coherent openings but loses
knowledge and the ability to stop. It is far below 9B (6/60 vs 51/60, loose). Reallocating the budget does not help.
Getting closer would need training (distillation on chat data), not weight selection.

# Rehearsal 5: light healing — how many tokens to reach 9B? (2026-11-09, design fixed before running)

## Question

The 81% cut (96×256, 8.5B total, 6/60 without training) lost knowledge and the ability to stop. Train it a little on
the teacher's own answers (sequence-level distillation from the original 35B-A3B). How many tokens bring it near
Qwen3.5-9B (51/60)? The goal: shrinking a model for a few hundred-thousandths of what training one from scratch costs.
(Qwen3 was pretrained on 36T tokens. 10M tokens is 1/3,600,000 of that.)

## Setup (`heal/`)

- **Student**: chat-picked 96×256 (`heal/kaggle/build`, cut on CPU from `stats.npz`). It runs in float16 on 2× Kaggle T4,
  split by layer. On the Mac check, float16 KL on A was 0.2896 against 0.2899 for float32, with no inf.
- **LoRA**: r 16, α 32, on attention, GDN (`in_proj_qkv`, `in_proj_z`, `out_proj`) and shared-expert linears.
  Routed experts and routers stay fixed (`lin` in rehearsal 3 got most of the gain).
- **Data**: what the student learns to write is the teacher's completion after `<think>\n`, through `<|im_end|>`.
  - R: answers from the original model (Q4_K_M GGUF, llama.cpp on 2× T4; `heal/kaggle/teacher`).
    - Questions: 9,000 in `heal/prompts.jsonl` (`heal/build_prompts.py`), 3,000 each:
      - math: MATH train (levels 3–5) plus GSM8K train
      - sci: NVIDIA OpenScience, 10 options
      - code: MBPP plus Magicoder-OSS (Python)
    - None come from the benchmark sources. Any question that shares a 13-word sequence with a benchmark question is
      dropped; prompt templates are excluded from that check.
    - Settings: thinking on, temperature 0.6, top-p 0.95, top-k 20, up to 8,192 tokens.
    - Only answers that close the thinking and stop on their own are kept. Correctness is not filtered (pure distillation).
  - G: the original model's own UltraChat answers (`jiapingW/qwen3.5-35b-a3b-ultrachat-regen`, first turn, thinking closed).
  - G makes up 30% of the tokens, interleaved so every milestone sees the same mix.
- **Training** (`heal/heal_train.py`, `heal/kaggle/train`):
  - Cross-entropy on completion tokens.
  - 32k trained tokens per step.
  - AdamW, learning rate 1e-4, constant after 10 warmup steps.
  - Dynamic loss scaling.
  - Sequences up to 9,216 tokens.
- **Milestones**: 1M, 3M and 10M tokens fed (prompts included). LoRA is saved at each one.

## Measurements

- KL to the teacher on the 60 held-out chat windows (A+B) at step 0 and each milestone.
- The 60 questions, run the same way as the 6/60 baseline:
  - Mac, 4-bit MLX student with the LoRA as an mlx-lm adapter
  - thinking on, 16k tokens
  - strict and loose grading, `think_closed`
  - code run on a disposable Kaggle VM
- Done at 10M, plus 1M and 3M if time allows.
- Cost = tokens fed + tokens the teacher generated, against 36T.

## Criteria

- **Reaches 9B**: loose ≥ 46/60 at some milestone (9B's 51 minus about 2 standard errors at n = 60).
- **Partly recovers**: loose ≥ 20/60 and `think_closed` ≥ 30/60.
- **Does not recover**: loose < 15/60.

## Known weaknesses

- The teacher answers come from a 4-bit teacher.
- The LoRA is trained on the float16 student but evaluated on the 4-bit one.
- One seed only.
- 60 questions is small.
- Teacher answers over 8,192 tokens are dropped, so the hardest problems are under-represented.

## Run log and deviations from the design (2026-11-09)

- **Teacher throughput**: llama.cpp (Q4_K_M, 16 slots on 2× T4) ran at about 118 completion tokens/s. In 7.1 h the
  teacher answered 796 questions. Of these, 656 closed their thinking within 8,192 tokens, giving 1.89M tokens:
  - math 157 (634k tokens), sci 238 (879k), code 261 (373k)
  - Truncated at 8,192 tokens: math 41%, sci 10%, code 2%
  - Of the kept answers, 137/157 math (87%) and 217/238 sci (91%) match the reference answer exactly.
- **Deviation**: there are fewer teacher answers than planned (1.9M instead of ~7M). So UltraChat answers make up 50% of
  the tokens instead of 30% (`--g-share 0.5`, about 3.8M tokens in total).
- **Training speed** (smoke test, float16, 2× T4):
  - About 206 tokens/s, with peak memory 12.1 GB per GPU at 9,216 tokens.
  - That needed chunked attention: with head_dim 256, SDPA on a T4 falls back to the math kernel and builds the full
    score matrix, 5 GB at 9k tokens.
  - KL at step 0 is 0.258, the same as the float32 pass.
- **Evaluation speed**: vLLM 0.33 on 2× T4 does run the student, but the untrained student's 60 answers (mostly loops to
  16k) took over 5 h. The Mac (4-bit, 4 at a time) took about 2.2 h. So milestones are scored on the Mac. A T4 vLLM run
  is kept for the final check.
- Kaggle allows two GPU sessions at a time, and outputs are only readable after a session ends. Training therefore runs
  in sessions that stop at milestones and resume from `state.pt`.

### First result: 1M tokens (2026-11-09, Mac 4-bit + LoRA adapter, 60 questions, 16k tokens)

| model | sci | math | code | loose | strict | `think_closed` |
|---|---:|---:|---:|---:|---:|---:|
| 96×256, untrained | 0 | 0 | 6 | 6/60 | 1/60 | 2/60 |
| 96×256 + LoRA, 1M tokens (KL 0.258 → 0.220) | 8 | 3 | 9 | **20/60** | 16/60 | 22/60 |
| Qwen3.5-9B | 19 | 16 | 16 | 51/60 | 48/60 | 54/60 |
| Qwen3.5-35B-A3B (teacher) | 20 | 19 | 20 | 59/60 | 57/60 | 57/60 |

- KL to the teacher: 0.258 at the start, 0.232 at 0.3M tokens, 0.220 at 1M and 0.209 at 3M.
- The answers that remain loop the way the untrained model did. The working is mostly right, then one slip (here "1.414 + 0.707 ≈ 1.125") starts endless re-checking, which ends in the same two lines repeated until 16k.
- **4-bit hurts this model**: on the vLLM T4 baseline in float16, the untrained student stopped after under 2k tokens on 5 code questions where the 4-bit Mac copy looped to 16k. So the Mac scores are likely a lower bound.
- **Variant with LoRA on the routed experts too** (rank 8, 169M parameters): KL 0.2249 against 0.2322 at 0.3M tokens. That closes 29% more of the gap per token, but trains at half the speed.
- **Loop-exit phase** (`--loop-exit`): every teacher answer again, with a stretch of its own thinking repeated 3–8 times. A break-out sentence follows ("I'm going in circles here…"), then the rest of the answer. The loss covers only the break-out and what comes after it.

### 4M and 6M (2026-11-09)

- 4M tokens (end of the teacher data at a 50% UltraChat share): KL 0.208. Mac 4-bit scores: sci 9, math 5, code 9,
  for **23/60 loose** (18/60 strict, `think_closed` 26/60). KL is nearly flat from 3M (0.209), so more of the same data
  does not help much.
- Loop-exit phase from 4M to 6M (session 3): KL 0.207 at 5M and 0.208 at 6M. The loss fell from 0.99 to 0.89. Its goal
  is breaking out of loops, which KL does not measure. Scored on the 60 questions on the Mac (4-bit) and on 2× T4 in
  float16 (vLLM, 8 h timeout per answer).
- Expert-LoRA variant: KL 0.212 at 1M, about where the linear-only run is at 3–4M. It is continued from 1M straight
  into the loop-exit phase (`--jump-to-loop-exit`) up to 2.5M.

### Loop-exit result (6M) and the expert-LoRA run (2026-11-10)

| model | KL (fp16) | sci | math | code | `think_closed` |
|---|---:|---:|---:|---:|---:|
| 4M (no loop-exit) | 0.208 | 9 | 5 | 9 | 26/60 |
| 6M (4M + 2M of the loop-exit phase) | 0.208 | **11** (strict 10) | **7** (strict 6) | 14 | **39/60** (code 19, sci 14, math 6) |
| expert LoRA, 1M plus 1.5M loop-exit (2.5M in total) | **0.203** | 10 (strict 10) | **8** (strict 7) | 14 | 38/60 (code 18, sci 12, math 8) |

(Mac 4-bit, loose grading)

- Loop-exit works on loops that repeat the same lines: answers that close their thinking go from 26 to 39 at the same
  KL.
- The math answers that still run to 16k are a different kind of loop. They wander between conflicting calculations
  (e.g. "45+46+47 = 138" against "3×47 = 141"), so the synthetic exact-repeat examples do not cover them.
- The expert-LoRA adapter converts to mlx (gate_up is split into gate/up; rank 8 is zero-padded to 16). Its Mac KL is
  0.224 against 0.203 in float16, the same 4-bit offset as the linear-only run (0.208 → 0.229).

## Result of rehearsal 5 (2026-11-10)

Everything below uses the same 60 questions, thinking on, up to 16k tokens. "loose" also takes the last answer inside
the thinking.

| model | where | sci | math | code | loose | strict | `think_closed` |
|---|---|---:|---:|---:|---:|---:|---:|
| 96×256, untrained | Mac 4-bit | 0 | 0 | 6 | 6/60 | 1/60 | 2/60 |
| + LoRA, 1M tokens | Mac 4-bit | 8 | 3 | 9 | 20/60 | 16/60 | 22/60 |
| + LoRA, 4M tokens | Mac 4-bit | 9 | 5 | 9 | 23/60 | 18/60 | 26/60 |
| + loop-exit, 6M tokens | Mac 4-bit | 11 | 7 | 14 | **32/60** | 30/60 | 39/60 |
| expert LoRA + loop-exit, 2.5M tokens | Mac 4-bit | 10 | 8 | 14 | **32/60** | 31/60 | 38/60 |
| + loop-exit, 6M tokens | **2× T4, float16, vLLM, seed 0** | 10 | 8 | 11 | **29/60** | 28/60 | 43/60 |
| Qwen3.5-9B | reference | 19 | 16 | 16 | 51/60 | 48/60 | 54/60 |
| Qwen3.5-4B (half the size of the cut model) | reference | 16 | 16 | 18 | 50/60 | 47/60 | 49/60 |
| Qwen3.5-35B-A3B (teacher) | reference | 20 | 19 | 20 | 59/60 | 57/60 | 57/60 |

- **Verdict against the criteria**: "partly recovers" (loose ≥ 20 and `think_closed` ≥ 30). It does not reach 9B (the
  target was ≥ 46). The float16 T4 run (29/60) is within noise of the Mac runs (32/60). It is the accurate number.
- **Where this really stands**: the healed 8.5B model (29–32/60) is far below Qwen3.5-4B (50/60), a ready-made model
  of half its size, on the same 60 questions (`results/xmodel_q35a3b/width/mac_96x256_grades.txt`). 4B and 9B are one
  question apart, so these 60 questions cannot tell "reached 9B" from "reached 4B". At an 81% cut with this recipe,
  cutting the big model is worse than taking the small one.
- **Cost**: 6.0M tokens fed to the student, plus 3.0M tokens generated by the teacher (all of them, including truncated
  answers), plus ~3M tokens of the teacher's public UltraChat answers. That is about 12M tokens, roughly 1/3,000,000 of
  Qwen3's 36T pretraining tokens. Compute was free Kaggle T4s, about 30 GPU-hours, half of them spent on slow
  generation and evaluation.
- **What moved the score**:
  1. Distilling the teacher's own answers into LoRA (6 → 23).
  2. Loop-exit training (23 → 32): it teaches the model to break out once its own lines start repeating.
  3. LoRA on the routed experts (rank 8): the same score with fewer than half the tokens (KL 0.203 at 2.5M against
     0.208 at 6M).
- **What is left**: math (7–8/20 against 16 for 9B). The remaining failures come from an arithmetic slip, then
  wandering between the conflicting results without repeating lines. Loop-exit on exact repeats does not cover this.
  Candidates:
  - commit training from the student's own wandering prefixes into the teacher's conclusion
  - more math teacher data with a longer limit than 8,192 tokens (41% of math answers were cut and dropped)
  - expert LoRA on more data
- Artifacts:
  - LoRAs: `cache/heal_run/{s1,s2,s3,exp1,exp2}/heal/`
  - mlx adapters: `cache/heal_adapters/`
  - answers and grades: `cache/heal_eval/`, `cache/heal_grades/`, `cache/heal_bench/`

# Rehearsal 6: checks on the weak points (2026-11-10, design fixed before running)

A review of everything above listed these weak points: the healed model loses to Qwen3.5-4B, only one keep ratio was
tried, the same 60 questions were reused for every decision, the old negative results (router bias, merging) were made
on the Wikipedia-picked cut, the sampler was never tuned, and nothing was committed. This section fixes what is run
and how it is read, before any of it has a result. Free compute only: the Mac (M5, 32 GB, 4-bit MLX), Kaggle CPU for
code grading, and what is left of the weekly Kaggle T4 quota (about 3 hours).

## 6a. Keep-count curve

- **Question**: is there a size at which the cut model is level with 4B or 9B without training?
- **KL** (`moe/width_pass.py`, float32, Mac MPS, the held-out chat windows A and B of `chat_run`): experts picked by
  chat routed mass as before, shapes 96/128/160/192/256 × 256 and 96/128/160/192 × 512. Sizes: 8.5B, 10.5B, 12.5B,
  14.5B, 18.5B (× 256) and 14.5B, 18.5B, 22.6B, 26.6B (× 512). Reference on the same ruler: 9B 0.119, 4B 0.143.
- **60 questions** (Mac 4-bit, untrained, same settings as before) at two points: 128×256, and the smallest shape whose
  KL is at or below 4B's 0.143 (if none is, the shape with the lowest KL among those up to 14.5B).
- **Reading**: "level with 4B" = loose ≥ 43 (within 7 of 50, the rough noise of 60 questions). KL alone decides nothing;
  rehearsal 5 moved from 23 to 32 at the same KL.

## 6b. A fresh check set and three samples

- **Set**: `fix_task/check_set.jsonl`, order 100–119 of each domain in `bench/tasks.jsonl` (60 questions). Orders 0–19 are
  the usual 60; orders 20–99 fed the expert choice and the KL ruler through the teacher's answers; the healing data was
  filtered against all of `bench/tasks.jsonl`. So nothing has seen 100–119.
- **Runs**: the 6M loop-exit LoRA and the expert-LoRA 2.5M model, three independent samples each (the mlx-lm server takes
  no per-request seed when batching, so a "seed" is a separate run), plus 9B and 4B once (4-bit MLX).
- **Reading**: two models differ only if the gap in mean loose score is larger than the spread of the three samples of
  either. The 32 vs 32 tie and the 29 vs 32 gap are judged by this.

## 6c. Sampler settings instead of loop-exit training

- **Question**: how much of loop-exit's +9 (23 → 32) could a setting have bought?
- **Settings**, each on the usual 60 questions, everything else unchanged:
  - repetition penalty 1.1 over the last 256 tokens
  - presence penalty 1.5 (Qwen's own advice against endless repetition)
  - DRY (multiplier 0.8, base 1.75, allowed length 2, whole context; `heal/mlx_dry_server.py`)
  - a thinking cap: cut the thinking of the existing answers at 4,096 or 8,192 tokens, close `</think>`, and let the
    model write the answer (`heal/think_cap.py`). Reuses the stored answers, so it is cheap.
- **Models**: the 4M LoRA (before loop-exit; the main comparison), the untrained cut (DRY and the cap), and the 6M
  loop-exit model (the cap, and the best setting found on 4M).
- **Reading**: a setting "replaces" loop-exit if 4M + setting ≥ 29 loose (within 3 of 32). Math and code are reported
  separately, because penalties on repeated tokens can hurt digits and code.

## 6d. Router bias and least-squares merging on the chat-picked cut

- Rehearsals 1 and 2 were run on the Wikipedia-picked 48 experts (chat KL 1.415). Redo `grad` router bias and `lsq`
  merging on the chat-picked 48×512 (0.325), same scripts, same `closed` on window set B. H1 and H3 stay as they were:
  `closed` ≥ 0.10 with the 95% interval above 0.

## 6e. Teacher signal (needs the GPU)

- Math answers again with a 16,384-token limit (`heal/kaggle/teacher2`, vLLM, GPTQ-Int4; already running).
- Score every stored teacher answer once more with the teacher and keep its top-20 probabilities at each position, so
  the same text carries a distribution instead of one token. Training then uses CE on the token plus KL on the top-20.
- Recovery data: let the student answer training questions, cut where its answer leaves the teacher's (first wrong
  intermediate result or first loop), and let the teacher continue from there. Train on the continuation only.
- With about 3 GPU-hours left this week, only one training run fits. Order: the top-up math data first (it exists),
  the two others after the weekly reset.

## 6f. WeightFix (the 235-weight task)

- Middle tasks between "one huge outlier" (solved by a 3-line rule) and "under-trained" (no signal): (1) a decoy —
  one large but harmless value next to the harmful one, (2) a sign flip of an ordinary-sized weight.
- Ranking instead of writing: show the LLM a net and k candidate edits from the greedy teacher's search, and ask which
  helps most. One net gives about 4,000 scored edits, so the labels are plentiful and not arbitrary.
- Both need a short LoRA training on a GPU. Data and baselines are built on CPU first.


# Rehearsal 6: results (2026-11-10)

## 6a. Keep-count curve (Mac MPS, float32, 2.8 h; `cache/keep_curve/width_results.json`)

KL to the original on the same 60 held-out chat windows; no training. `full` gives 0.0016 again.

| experts × neurons | size | KL | chat mass kept | routed-output error |
|---|---:|---:|---:|---:|
| 96×256 (the cut so far) | 8.5B | 0.268 | 70% | 0.53 |
| 128×256 | 10.5B | 0.214 | 81% | 0.49 |
| 160×256 | 12.5B | 0.187 | 89% | 0.45 |
| 192×256 | 14.5B | 0.171 | 95% | 0.43 |
| 256×256 (every expert, half width) | 18.5B | 0.159 | 100% | 0.40 |
| 96×512 (full width) | 14.5B | **0.113** | 70% | 0.43 |
| 128×512 | 18.5B | 0.059 | 81% | 0.35 |
| 160×512 | 22.6B | 0.031 | 89% | 0.28 |
| 192×512 | 26.6B | 0.015 | 95% | 0.21 |
| Qwen3.5-9B / 4B (trained) | | 0.119 / 0.143 | | |

- **Halving the width costs more than dropping experts.** With every expert kept, half width alone leaves 0.159 —
  no ×256 shape of any size reaches 4B's 0.143. At full width, 96 experts (14.5B) are already at 0.113, below 9B.
- At the same 14.5B, 96×512 beats 192×256 (0.113 vs 0.171). The 8.5B budget is where the two effects cross
  (96×256 0.267 vs 48×512 0.325); above it, spend parameters on width first.
- By the rule set in the design, the two shapes to score on 60 questions are **128×256** and **96×512**.
- Fixed on the way: `width_pass.py` joined `--out` to the run directory and failed at the last line when given a
  path; the table above was rescued from the log.

### 60 questions at 96×512, untrained (14.5B, Mac 4-bit; 4-bit KL 0.130, 128×256 0.232, 96×256 0.288)

Loose score (science / math / code), the usual 60 questions, no training at all. The caps are the two fixed in 6c.

| setting | loose | strict | truncated answers | tokens per question |
|---|---:|---:|---:|---:|
| no cap (16,000-token limit) | 37 (9 / 9 / 19) | 24 | 34 of 60 | 10,397 |
| thinking cap 8,192 | **48** (12 / 19 / 17) | 49 | 8 of 60 | 6,484 |
| thinking cap 4,096 | 45 (10 / 18 / 17) | 45 | 13 of 60 | 3,954 |
| Qwen3.5-9B / 4B (no cap) | 51 / 50 | | | |

- **The wide cut has not lost the knowledge; it has lost the ability to stop thinking.** Without a cap 18 of 20
  science answers and 13 of 20 math answers run to the limit. Math goes 9 → 19 with a cap: read by hand, the raw
  answers reach the right result early and then re-check or repeat one line ("I will output 23.") until the limit;
  the capped answers are complete worked solutions written by the model itself (the cap script never sees a reference).
- This is the opposite of 96×256, where math stays at 6–7 with or without the cap (6c): there the result is not there.
- Science is what is left (12 of 20): 7 of the 8 wrong answers loop inside the final answer too and run out of the
  2,048 answer tokens. A stopping rule for the answer, or loop-exit training, is the next lever there.
- The mlx-lm server died 7 times on this model's long loops (Metal resources); `mac_eval.sh` restarts and resumes.
- **Caveat**: the 9B and 4B numbers are without a cap, and the caps were chosen before this run but the 60 questions
  are the usual ones. The check set (6b) repeats all of them with the same caps.
- **128×256 untrained (10.5B)** on the same 60 questions: 17 (2 / 2 / 13), 51 of 60 answers run to the limit and 49
  loop. More experts at half width do not make a usable model (96×256 was 6); this agrees with the KL curve.

## 6b. The fresh check set, every model with the same caps (Mac 4-bit; one sample each so far)

`fix_task/check_set.jsonl` (orders 100–119, seen by nothing). Loose score (science / math / code); code graded on a
Kaggle CPU VM. The set is harder than the usual 60: 9B drops from 51 to 43, and 9B and 4B themselves run out of the
16,000 tokens on 11 and 13 of the 20 math questions.

| model | size | no cap | cap 4,096 | cap 8,192 |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | 9B | 43 (16 / 10 / 17) | 40 (13 / 14 / 13) | 46 (15 / 16 / 15) |
| Qwen3.5-4B | 4B | 39 (13 / 10 / 16) | 45 (14 / 13 / 18) | 44 (13 / 15 / 16) |
| 96×512, untrained | 14.5B | 30 (7 / 10 / 13) | 37 (9 / 15 / 13) | **43** (11 / 20 / 12) |
| 96×256, 6M loop-exit LoRA | 8.5B | 28 (9 / 9 / 10) | 29 (11 / 8 / 10) | 32 (11 / 11 / 10) |

- **The 96×512 result holds on unseen questions, as a relative statement.** With the 8,192 cap the untrained 14.5B
  cut is 3 below 9B and 1 below 4B (43 / 46 / 44), inside the noise of 60 questions. Without a cap it is 9–13 below,
  with the 4,096 cap 3–8 below: it needs more thinking room than the trained small models before it has the result.
- The cap helps the trained small models on math too (10 → 14–16): a thinking cap is a fair setting for everyone,
  not a crutch for the cut model only. So the usual-60 comparison "48 with a cap against 51 without" is not like for
  like (9B and 4B with a cap were not measured there); this table is the one to read.
- Per domain at cap 8,192: math 20 of 20 (9B 16, 4B 15; answers read by hand, they are the model's own solutions),
  science 11 (15, 13), code 12 (15, 16). Science and code are where the cut still loses: 4 science and 4 code
  answers loop inside the final answer even after the cap.
- The healed 96×256 model is 11–14 below all of them at cap 8,192 (32), about the same gap as on the usual 60.
- Still running: two more samples of 96×512 and of the 6M model, and three of the expert-LoRA model, for the spread.
- Fixed on the way: `mac_think_cap.sh` stopped at the first server crash and the queue moved on without the cap
  files; it now restarts the server and resumes from the `.partial` file.

## 6c. A thinking cap instead of loop-exit training (Mac 4-bit, the stored 60-question answers; `heal/think_cap.py`)

Answers that thought for more than the cap are cut there, `</think>` is closed, and the model writes only the final
answer (up to 2,048 tokens). No training, no new sampling of the thinking. Loose score (science / math / code):

| model | no cap | cap 8,192 | cap 4,096 | tokens per question (no cap → 4,096) |
|---|---:|---:|---:|---:|
| 4M (before loop-exit) | 23 (9 / 5 / 9) | 28 (10 / 6 / 12) | 28 (9 / 7 / 12) | 10,579 → 4,221 |
| 6M (after loop-exit) | 32 (11 / 7 / 14) | 33 (12 / 7 / 14) | 32 (12 / 7 / 13) | 8,091 → 3,640 |

- On the 4M model the cap recovers 5 of the 9 points that loop-exit training gave (28 against the pre-set bar of 29:
  one short, inside the noise of 60 questions). Most of the gain is code (9 → 12): answers that had the right code
  somewhere but never stopped. On the 6M model the cap adds nothing more.
- With a cap every answer finishes, so strict = loose (4M science + math strict 9 → 16, 6M all three 30 → 32–33) at well under half the tokens.
- **Math does not move** (5–7 of 20 with or without the cap): the model does not have the right result by token 4,096
  or 8,192 either. That part needs training signal, not a stopping rule.
- Fixed on the way: `mac_eval.sh` and `mac_think_cap.sh` failed without an adapter (empty array under `set -u` on
  macOS bash 3.2).

## 6g. A loop watchdog on the thinking (design written 2026-11-11 after 6b, before any run)

- **Why**: on the check set the untrained 96×512 is at 43 with the 8,192 cap, and what is left is tied to loops in
  the thinking. With that cap, answers whose thinking loops are right 9 of 18 (science) and 2 of 7 (code); answers
  whose thinking does not loop are right 2 of 2 and 10 of 13. The loops start early (science: half of them before
  about 4,300 tokens), so a cap at 8,192 hands the model thousands of tokens of loop before the answer. The loops
  begin when the model fails to spell a term ("ammonium nitate", "Acic acid", "Arg" for argon), decides the question
  has a typo and circles around it: the dropped experts took rare subwords with them.
- **Setting** (`think_cap.py --loop-cut`): cut the thinking where it starts to repeat — the first 4,000-character
  window that compresses below 15% with zlib, the grader's own loop rule, not tuned — or at 8,192 tokens, whichever
  comes first; close `</think>`; the model writes the answer. Variant: one fixed sentence after a loop cut
  ("I am going in circles. I will stop here and give my best answer now."). A server could do the same while streaming.
- **Protocol** (no tuning on what is reported): the two variants are compared on the check set (now a development
  set) for 96×512, 9B and 4B. One variant is picked there, then run once on a **second fresh set**
  (`fix_task/check_set2.jsonl`, orders 120–139, seen by nothing) and on the usual 60, for 96×512, 9B and 4B alike.
- **Reading**: the watchdog "helps" if 96×512 gains at least 3 on the check set; the goal of 46 is "reached" only if
  the second fresh set agrees (96×512 within 3 of 9B there with the same setting).

### Result on the check set (2026-11-11; loose, science / math / code; `cache/heal_grades/loop1/`)

| model | cap 8,192 (6b) | + loop cut | + loop cut + exit sentence | answers cut at a loop |
|---|---|---|---|---|
| 96×512 untrained | 43 (11 / 20 / 12) | 41 (12 / 17 / 12) | 42 (11 / 18 / 13) | 23 of 60 |
| 9B | 46 (15 / 16 / 15) | 45 (13 / 16 / 16) | 45 (13 / 15 / 17) | 0 |
| 4B | 44 (13 / 15 / 16) | 43 (13 / 13 / 17) | 43 (13 / 13 / 17) | 7 |

- **The watchdog does not help** (needed +3, got −2 and −1), so it is dropped. Cutting the loop away earlier does not
  bring the answer back: when the model cannot spell the term it needs, the answer written after the cut is wrong as
  well. Math lost 2–3 because some looping answers already held the right result further down, past the cut.
- 9B has no loop cut at all, so its two extra columns are the same thinking with the answer written again: that alone
  moves the total by 1 and a domain by up to 2. Differences of 1–2 between these columns are noise.
- The setting carried to the second fresh set is therefore the plain 8,192 cap, for all three models.
- Usual 60, 128×256 untrained with caps: 17 without a cap → 22 (cap 4,096: 5 / 5 / 12), 24 (cap 8,192: 7 / 5 / 12).
  The cap does not rescue a model whose answers loop too.

## 6h. Uneven widths inside 9B's size (design written 2026-11-11, before the run)

- **Why**: the author rules out 96×512 — at 14.5B it is 1.5 times 9B, and the point of the cut is to compare with 9B at
  9B's size. So the question goes back to the 8.5B budget (24,576 routed neurons per layer), where every shape tried
  so far gave all its experts the same width.
- **Idea**: keep the budget's most important neurons wherever they sit, so heavily used experts stay wide and rarely
  used ones become narrow (`moe/width_pass.py`, shapes `g…` and `t…`; importance and routed mass are the same chat
  statistics as before; no training). On the stored statistics (`cache/shape/stats.npz`), the uniform shapes trade
  one thing for the other and the uneven ones get both:

  | shape (24,576 neurons per layer) | experts kept | importance kept | routed mass kept | KL (known) |
  |---|---:|---:|---:|---:|
  | 48×512 | 48 | 53% | 47% | 0.325 |
  | 96×256 | 96 | 46% | 70% | 0.267 |
  | 128×192 | 128 | 41% | 80% | 0.317 |
  | `g96`: best neurons among the top 96 experts | 89 | 57% | 67% | ? |
  | `g128`: … among the top 128 | 110 | 58% | 74% | ? |
  | `g256`: … among all | 140 | 58% | 81% | ? |
  | `g256f64`: same, experts under 64 neurons dropped | 69 | 58% | 56% | ? |
  | `t32x512+64x128`: top 32 at full width, next 64 at a quarter | 96 | 54% | 70% | ? |

  These two columns did not predict KL among the uniform shapes, so only the KL pass can tell.
- **Run**: one KL pass on the Mac (float32, the same 60 held-out chat windows): `96x256` again as the reference, the
  five uneven shapes above, and at 9B's full size (9.4B = 28,160 neurons per layer) `110x256` and `g128b28160`.
- **Reading** (fixed now): an uneven shape "wins" if its KL is below 96×256 with the whole bootstrap interval of the
  difference on its side. Reference points: 4B 0.143, 9B 0.119. If the best one wins, it is built as a model
  (stored padded for now; parameters counted as kept neurons), answers the check set with the 8,192 cap, and is the
  shape to heal. If none wins, uniform 96×256 stays and the remaining lever at this size is training.

### Result (2026-11-12, Mac MPS float32, 1.6 h; `cache/uneven/width_results.json`)

| shape | total | experts per layer (median width) | KL | gap to 96×256 closed [95%] |
|---|---:|---:|---:|---:|
| 96×256 (reference) | 8.5B | 96 (256) | 0.268 | — |
| `t32x512+64x128` | 8.5B | 96 (128) | 0.252 | 6% [−1, 13] |
| `g96` | 8.5B | 89 (297) | 0.193 | 28% [22, 35] |
| `g256f64` | 8.5B | 69 (421) | 0.184 | 31% [26, 37] |
| `g128` | 8.5B | 110 (164) | **0.182** | 32% [26, 39] |
| `g256` | 8.5B | 140 (70) | **0.182** | 32% [25, 39] |
| 110×256 | 9.4B | 110 (256) | 0.237 | 12% [10, 13] |
| `g128b28160` | 9.4B | 114 (225) | **0.155** | 42% [37, 48] |

- **Uneven widths win** by the rule above. At the same 8.5B, picking neurons across experts takes KL from 0.268 to
  0.182, below uniform 128×256 at 10.5B (0.214). Two fixed tiers do not (interval includes 0): the gain comes from
  letting every expert have its own width, not from "a few wide, the rest narrow".
- Which experts are in the pool matters little once the pool is 128 or more (`g128` = `g256`), and dropping the thin
  experts costs almost nothing (`g256f64`: 69 experts, 0.184).
- At 9B's own size (9.4B), `g128b28160` reaches 0.155: near 4B (0.143), with 9B at 0.119, and the same as uniform
  256×256 (0.159), which needs 18.5B.
- The first try of this pass filled the Mac's swap (layer time 170 s → 630 s and rising): the uneven shapes change
  size at every layer, so MPS never reused its cached blocks. Fixed by returning them after every shape
  (`free_cache`); the pass then ran at a steady 150 s per layer.
- KL is not the benchmark: 96×512 (0.113) looked better than 9B on KL and scored below it. What this buys in answers
  is the next measurement.

### Next (design written 2026-11-12, before any of it ran)

- **Shape to carry on with**: `g128b28160` (9.4B, exactly 9B's parameter count; the best KL inside that size).
  `g128` (8.5B) is built too for the Mac, as the same-budget comparison with 96×256.
- **Storage**: `moe/uneven.py` keeps the experts back to back at their own widths (no padding; `build_shape.py`
  writes it, `heal_train.py` loads it; a tiny-model check is in `moe/smoke_uneven.py`). For the Mac the experts are
  stored at full width with the dropped neurons' gate and up rows zeroed (exactly zero after 4-bit, so the output is
  the same as the unpadded model's): simple, but it runs at the original model's speed, not the cut model's.
- **No-training test (Mac)**: the 6M LoRA of 96×256 touches attention, GDN and the shared expert only, so it loads
  on any shape. KL of each new shape with and without it; then the check set with the 8,192 cap for `g128b28160`
  with it. This LoRA was fitted to another cut, so this is a lower bound for the shape.
- **Training (Kaggle T4×2, the 1 h 52 min left this week)**: `g128b28160`, started from the 6M LoRA with a new
  optimizer (`--init-lora`), about 1.1M tokens at 243 tokens/s. Data: all teacher answers shuffled together
  (`--mix-traces`: the rehearsal 5 answers plus the 253 right math top-up answers of 6e), UltraChat at 30% of the
  tokens; no loop-exit examples (the LoRA already has them; `think_closed` afterwards shows whether they were kept).
  The session stops itself after 1.2 h and saves the LoRA with its KL.
- **Reading**: the check set with the 8,192 cap, against healed 96×256 (32), 4B (44) and 9B (46). The comparison with
  healed 96×256 mixes three things (shape, 0.9B more parameters, 1.1M more tokens with new math answers); the
  no-training test above separates the shape. There is no quota for a 96×256 control with the same extra tokens.

### `g128b28160`: KL with the old LoRA and after the Kaggle session (2026-11-12)

| model | KL, Mac 4-bit | KL, T4 float16 |
|---|---:|---:|
| 96×256, untrained | 0.288 | 0.258 |
| 96×256 + its 6M LoRA | 0.229 | 0.208 |
| `g128b28160`, untrained | 0.172 | 0.148 |
| `g128b28160` + the 6M LoRA of 96×256 | 0.166 | 0.146 |
| `g128b28160` + that LoRA + 0.56M tokens on this shape | (below) | **0.138** |

- The builds agree: the Mac 4-bit model sits 0.02 above the float32 pass (0.155), the same offset as 96×256, and the
  Kaggle float16 student gives 0.148 before any LoRA (the session's own sanity check, limit 0.3).
- The LoRA fitted to 96×256 neither breaks the new shape nor helps it much (0.148 → 0.146); 15 optimizer steps on the
  new shape move it to 0.138. Untrained, this shape is already closer to the teacher than healed 96×256.
- **The session was half as fast as planned**: 128 tokens/s against 243 for 96×256, so 0.56M tokens in 1.35 h, not
  1.1M (92 min of GPU in all; the quota is now used up). transformers runs the stock experts through a batched
  implementation on GPU; `RaggedExperts` is a plain loop over the experts (on CPU the two take the same time). A
  bucketed, batched version is the fix before any longer run. One 9k-token sequence was skipped (loss scale), none
  for memory (peak 12.5 GB of 15).
- Files: `cache/heal_run/u1/heal/` (LoRA, log), `cache/heal_adapters/g128b_t1` (MLX adapter, made by mapping the
  tensors onto the 6M adapter's keys — the mapping reproduces that adapter exactly).

### `g128b28160` on the check set (2026-11-12; Mac 4-bit, one sample each, loose; science / math / code)

| model | no cap (16k) | thinking cap 8,192 | closes its thinking by itself |
|---|---:|---:|---:|
| 96×256 8.5B + its 6M LoRA | 28 (9 / 9 / 10) | 32 (11 / 11 / 10) | 37 of 60 |
| **`g128b28160` 9.4B + the 6M LoRA of 96×256** | **43** (14 / 18 / 11) | **43** (13 / 19 / 11) | 53 of 60 |
| `g128b28160` + that LoRA + 0.56M tokens (Kaggle) | 39 (11 / 15 / 13) | **43** (14 / 16 / 13) | 41 of 60 |
| Qwen3.5-4B | — | 44 (13 / 15 / 16) | 36 of 60 |
| Qwen3.5-9B | — | 46 (15 / 16 / 15) | 41 of 60 |

- **The shape is worth 11 questions** with the same LoRA and the same cap (32 → 43; 0.9B more parameters are part
  of it). It is one question from 4B and three from 9B; one sample each, so 43, 44 and 46 are not told apart.
- Math is the best of all five (19); **code is what is missing** (11–13 against 15–16): 19 of 20 code answers finish
  and 8–9 of them are wrong. The same picture as 6i, at a higher level.
- **The Kaggle session did not raise the score.** KL improved (4-bit 0.166 → 0.157), code went 11 → 13, but the
  model stopped closing its thinking as often (53 → 41 of 60), so without the cap it loses four questions and with
  it the total is the same. The session had no loop-exit examples and half of its teacher tokens were the long math
  top-up answers (median 5.5k tokens): the risk named in the design above. With the two LoRAs, science + math agree
  on 34 of 40 questions (28 both right, 6 both wrong; 4 only the old one, 2 only the new one).
- **Two failures of the harness on the way** (no result lost, about 11 hours of Mac time):
  - 06:26, the Mac rebooted (watchdog panic). The server of this 8.8 GB model grew to 16.5 GB during the cap run;
    mlx-lm wires its memory, and wired memory reached 30 of 32 GB. Servers now start through
    `heal/mlx_safe_server.py` (wired limit 11 GB, 1 GB buffer cache, small prompt cache) with `heal/mem_guard.sh`
    beside them.
  - 07:50, the guard killed a server (limit too low at 14 GB with 39% of memory free) and `mac_eval.sh` then exited
    on `kill` of the dead process under `set -e`; the queue went on to the baselines and the trained model's run
    sat at 18/60 until 17:00. Fixed (`|| true`; the guard now acts at 18 GB or under 10% free).

### Unseen sets (reading fixed 2026-11-12 20:10, before these runs)

- The two LoRAs tie on the check set, so no choice is made: both are run, the trained one first, on the usual 60 and
  on the second fresh set, each with the 8,192 cap (the condition every model is compared under). They are two
  samples of "this shape, lightly healed", not a selection.
- The goal of the rehearsal is stated on the usual 60: 46/60. Same Mac, same cap: 9B 52 (18 / 17 / 17), 4B 56
  (19 / 19 / 18); without the cap 9B 43, 4B 46. The second fresh set with the cap: 9B 45 (16 / 12 / 17), 4B 44
  (15 / 13 / 16); without it both 39.
- "Reaches 9B's level" needs both sets within the spread of one sample (about ±3) of 9B; reaching 46 on the usual 60
  alone would meet the letter of the goal while 9B and 4B sit at 52–56 there under the same cap.

- **Usual 60, `g128b28160` + trained LoRA** (2026-11-13; Mac 4-bit, one sample, loose; `cache/heal_grades/g128b_d/`):
  no cap 44 (12 / 14 / 18), closes its thinking in 46 of 60; cap 8,192: **50 (14 / 18 / 18)**. Against 9B 43 → 52 and
  4B 46 → 56. So 46 is passed with the cap and missed by two without it, and the model is 2 under 9B and 6 under 4B
  in the same condition. Code is 18 of 20 here (9B 17, 4B 18): the usual 60's code questions are easier than the
  check set's (13 of 20), so the code gap of the check set does not show on this set. Science is the weak domain
  here (14 against 18–19). The old LoRA's run and the second fresh set follow.

- **Usual 60, `g128b28160` + old LoRA** (2026-11-13; `cache/heal_grades/g128b_e/`): no cap 45 (17 / 15 / 13), closes
  its thinking in 49 of 60; cap 8,192: **50 (19 / 17 / 14)**. Both LoRAs land on 50 with the cap (9B 52, 4B 56), with
  the points in different places: the old LoRA is level with 4B and 9B on science and loses on code (14 against
  17–18, the check set's picture); the trained one is level on code and loses on science. One sample each, so the
  split by domain is within noise; the total (50 and 50, 44 and 45 without the cap) is the stable part.

- **Second fresh set, `g128b28160` + trained LoRA** (2026-11-13; `cache/heal_grades/g128b_f/`): no cap 40
  (12 / 11 / 17), closes its thinking in 38 of 60; cap 8,192: **45 (14 / 14 / 17)**. 9B: 39 → 45 (16 / 12 / 17),
  4B: 39 → 44 (15 / 13 / 16). With the usual 60 (50 against 52) this LoRA is inside ±3 of 9B on both unseen sets,
  which is the condition fixed above for "reaches 9B's level" (Mac 4-bit, cap 8,192, one sample per set). The old
  LoRA's run on this set is still to come; the claim is for the shape with light healing only if it holds too.

- **Second fresh set, `g128b28160` + old LoRA** (2026-11-13; `cache/heal_grades/g128b_g/`): no cap 44 (14 / 13 / 17),
  closes its thinking in 49 of 60; cap 8,192: **46 (14 / 15 / 17)**. 96×256 + old LoRA on the same set with the
  cap: 27 (12 / 5 / 10).

**Unseen sets, summary** (Mac 4-bit, thinking cap 8,192, one sample each, loose):

| model | params | usual 60 | second fresh set | check set (used for choices) |
|---|---|---|---|---|
| `g128b28160` + trained LoRA | 9.4B | 50 | 45 | 43 |
| `g128b28160` + old LoRA | 9.4B | 50 | 46 | 43 |
| 96×256 + old LoRA | 8.5B | — | 27 | 32 |
| Qwen3.5-9B | 9.4B | 52 | 45 | 46 |
| Qwen3.5-4B | 4B | 56 | 44 | 44 |

- By the reading fixed before these runs: both LoRAs are within ±3 of 9B on both unseen sets, so **the uneven shape
  at 9B's size, lightly healed, reaches 9B's level on these sets** — as a 73% cut of the 35B MoE that is still an
  MoE. The shape did it, not the training: the same LoRA on 96×256 gives 27 and 32, and the Kaggle session on the
  new shape changed nothing (50 / 45 against 50 / 46).
- What it does not say: (1) 4B, less than half the size, is level or ahead (56 on the usual 60), so the cut model
  is not yet worth its size; (2) the goal's 46 on the usual 60 is met only with the cap (44 and 45 without); (3)
  one sample per cell on a 4-bit Mac build; the exact T4×2 benchmark waits for the Kaggle quota; (4) the slips of
  names and words (below) are not in these scores' favour and are the open problem.

### Faster uneven experts for training (2026-11-13; `moe/uneven.py`, `moe/bench_uneven.py`)

- The Kaggle session trained at 128 tokens/s against 243 for 96×256. `RaggedExperts` was a plain loop: per expert
  a `where` (a GPU → CPU sync), two gathers, two linears and an `index_add_`, each a node for autograd.
- Now ("grouped", the default; the loop stays as `impl="loop"`): the (token, expert) pairs are sorted by expert
  once, each expert runs on a contiguous slice with `torch.mm(out=...)` into one padded buffer, and two functions
  with a hand-written backward replace the per-expert autograd nodes — the scheme of transformers' grouped experts
  forward, with a width per expert. One sync per layer instead of one per expert.
- Checked: outputs and gradients (input, routing weights, expert weights) agree with the loop to 1e-6 (float32);
  the tiny-model checks (`smoke_uneven.py`, `smoke_heal.py`) pass.
- Speed, one layer of the real size on the Mac's CPU (2,048 tokens, 113 experts): 1.5 times the loop with frozen
  experts, 16 times with trainable experts (the loop builds a full-size zero gradient per expert). **Not measured
  on a T4 yet**: how much of the 128 → 243 gap this closes is unknown until `bench_uneven.py --device cuda --dtype
  float16` runs at the start of the next Kaggle session (the mixed-in long math answers may be part of the gap).

### Why code lags: a greedy re-run (design written 2026-11-12 21:00, before the run)

- What is known (details in the result below): the cut models finish their code answers and fail in them; part of
  the failures are names that do not exist (`lfungey_cache` for `lru_cache`, `hashlib.newmd5`, `str.swap`).
- Question: is the right name still the model's most likely token and lost only when sampling (temperature 0.6 over
  a flattened distribution), or is the model's best guess itself wrong?
- Run: the 20 code questions of the check set again with temperature 0, `g128b28160` + the trained LoRA (the server
  that is up anyway), everything else unchanged. Graded and diagnosed on Kaggle like the sampled run (13/20, with
  2 crashes on names in finished answers).
- Reading: greedy ≥ 16/20 with the name crashes gone → the knowledge is there and the distribution is too flat (a
  sampler setting or light training can use it). The same failures under greedy → the best guess is wrong, and
  only putting neurons back or training the experts can fix it. In between: both.

### Why code lags: what was found (2026-11-12; check set, 20 code questions; `heal/kaggle/diag`, `heal/name_slips.py`)

How each code answer ends, run on a Kaggle CPU VM (`cache/heal_grades/diag1/`, `diag2/`; raw answers, no cap):

| answers | pass | wrong output | crash in a finished answer | unfinished |
|---|---|---|---|---|
| `g128b28160` + old LoRA | 11 | 5 | 3 | 1 |
| `g128b28160` + trained LoRA | 13 | 1 | 2 | 4 |
| 96×256 + old LoRA | 10 | 3 | 5 | 2 |
| 4B | 16 | 0 | 0 | 4 |
| 9B | 17 | 1 | 1 | 1 |
| teacher (usual 60) | 20 | 0 | 0 | 0 |

- **4B and 9B lose code questions by not finishing; the cut models finish and are wrong.** Their errors are slips of
  names and of consistency, not of the method: `from functools import lfungey_cache` and then `@functools.cache`;
  `expression = ...` and then `exp += ...`; `s_lower` made and `s` used; a comment that says "return the first list"
  above `return lst2`; `hashlib.newmd5`, `str.swap`, `from operator import operator`.
- Static count of names that do not exist (finished answers of every stored run, no execution): teacher 0 of 20,
  9B 0 of 53, 4B 0 of 47, cut models 7 of 71 (a lower bound; it does not see wrong logic).
- **KL does not see it.** KL on the held-out windows by kind (20 code, 38 math): 9B 0.211 / 0.077, 4B 0.247 / 0.095,
  `g128b28160` float32 untrained 0.210 / 0.133, 96×256 0.315 / 0.256, `g128b28160` + trained LoRA fp16 0.216 / 0.103.
  On code the new shape is level with 9B by KL and far behind by tests: a rare wrong token inside a name costs
  almost no KL and the whole answer.
- Not the cause: missing code in the calibration data (39 of the 120 windows are code, 78 math).
- **Overlap found**: 2 questions of the check set are inside the calibration windows (none of the usual 60, none of
  check set 2). They touch only the neuron choice, not any training; the check set's numbers above keep them.
- **Greedy re-run** (`g128b28160` + trained LoRA, temperature 0; the limit was 8,192 tokens, not 16,000): 12 of 20
  (sampled 13). No made-up name in the 14 finished answers (sampled: 2 in 16; `hashlib.md5(text)` without `encode`
  remains, a real function misused). But 6 answers did not finish (sampled 4): two that closed when sampled loop
  under greedy. By the reading fixed above this is "in between": the right name is usually still the most likely
  token, so sampling from a flattened distribution is part of the damage, but greedy is no fix because it loops more.
  One sample of 20, so the name count (0 against 2) is a hint only.
- What follows: (1) the slips are the target, not the method knowledge — training text should contain code and exact
  identifiers (the Kaggle session's top-up was math only); (2) a sampler between the two (lower temperature or top-p,
  loops watched) is a no-training candidate, to be chosen on the check set and confirmed on unseen sets; (3) the
  same slips show in prose, next section.

### A sampler between the usual one and greedy (design written 2026-11-13, before any run)

- Why: under greedy the made-up names went away and the loops grew. Is there a setting that keeps the first without
  the second? No training; the choice is made on the check set and then read once on the second fresh set.
- Candidates (two, fixed here): temperature 0.3 with top-p 0.95; temperature 0.6 with top-p 0.8. Top-k 20 and all
  else as usual. Model: `g128b28160` + trained LoRA. Full check set (60), raw and with the 8,192 cap (the answer
  after the cap uses the same sampler).
- What is read. A score moves by about ±3 between samples, so the score cannot pick a sampler by itself. The sharper
  numbers are the slips: private unknown words per 10,000 (now 70.4; `word_slips.py` against the same 4B and 9B
  files), code answers with a name that does not exist (now 2 of 16 finished), and how many answers close their
  thinking (now 41 of 60).
- Rule: a candidate is taken if its unknown-word rate is at most half the usual sampler's, it closes its thinking
  in at least 38 of 60, and its capped score is not under 40 (43 now). If both pass, the lower word rate wins. The
  winner is then run once on the second fresh set and reported next to the usual sampler's 45 there, whatever it
  gives. If none passes, the sampler stays as it is and that is the result.

### Made-up words in prose (design written 2026-11-12 21:50, before the count; `heal/word_slips.py`)

- Question (the author's): did the cut raise the hallucination rate? The made-up names in code are one form of it.
  A cheap first look, with no new run: do the stored science and math answers also hold more words that do not exist?
- Count: the check set's raw answers (no cap; thinking and answer; code blocks, formulas and the code questions left
  out), words of five letters or more. A word is "unknown" if it is not in the system dictionary (plain suffixes
  stripped) and not in any question. An unknown word is "private" if no other compared model writes it: technical
  terms are shared between models that answer the same questions, slips are not. The rule is the same for every model.
- Reported: private unknown words per 10,000 words, and the words themselves for reading (a count of spellings, not
  of false statements).
- Reading: a cut model at twice the rate of both 9B and 4B or more → prose is hit as well. Otherwise prose spelling
  is not visibly hurt, and only the probe with questions (things that do not exist; short facts) can tell.

### Made-up words in prose: result (2026-11-12; check set, 40 science and math answers each, no cap)

| answers | words | private unknown words per 10,000 | distinct | answers with one (final answer) |
|---|---|---|---|---|
| `g128b28160` + old LoRA | 35,617 | 39.9 | 56 | 28 of 40 (8) |
| `g128b28160` + trained LoRA | 53,562 | 70.4 | 59 | 27 of 40 (6) |
| 96×256 + old LoRA | 52,667 | 63.2 | 55 | 25 of 40 (3) |
| 4B | 53,809 | 7.6 | 29 | 18 of 40 (0) |
| 9B | 49,930 | 15.0 | 41 | 18 of 40 (2) |

- By the rule: every cut model is at 2.7 times 9B's rate or more, so **prose is hit as well**. The count understates
  it: 4B's and 9B's private words are real ones the dictionary lacks (`substring`, `deuterated`, `Medawar`, `Waals`,
  `Chebyshev`), the cut models' are not words: `interimolecular` (25 times), `octuminary`, `ququotant`,
  `incorincorrect`, `quadradratic`, `capacacity`, `concentcentration`, `avavotary`, `vanvanin`, `stomochemistry`,
  `acceluated`, `mathically`, `impimpance`.
- Two shapes of slip: a word piece written twice (`ququotant`, `incorincorrect`, `capacacity`) and a wrong piece in
  the middle of a word (`interimolecular`, `acceluated`, `stomochemistry`). Both are the code failures' shape
  (`lfungey_cache`, `hashlib.newmd5`): the model loses the place inside a multi-token word. In prose the reader
  still understands; in code one such slip fails the answer. The wider shape (`g128b28160`) does not slip less than
  96×256, and the Kaggle training did not help.
- Limits: one sample per model; the rate counts repeats (`argate` 250 times in one looping answer), the distinct
  count does not; this measures spelling, not whether statements are true. That needs the probe with questions.

## 6j. Does the cut make things up more? A probe with questions (design written 2026-11-13, before any run)

- Why: the cut models write names and words that do not exist (6h, code and prose). That counts spellings. This asks
  whether they also state false things more often than a ready-made model of the same size.
- Set (`heal/halluc_probe.py make` → `fix_task/halluc_set.jsonl`; written for this probe, in no benchmark set):
  40 questions about things that do not exist (library functions, laws, theorems, people, works, units; "What does
  … do?", no hint that it may not exist), 10 real things asked the same way (a model that denies everything fails
  these), 60 short facts with one right answer ("Give only the answer. If you do not know, say I don't know").
- Run: Mac 4-bit, the usual sampling, **thinking off**, up to 1,024 tokens, one sample: `g128b28160` with each LoRA,
  96×256 + old LoRA, 4B, 9B. The teacher does not fit on the Mac.
- Changed 2026-11-13 08:10 after a pilot, before any grading: the first plan was thinking on with 4,096 tokens. The
  first six answers of 9B (`cache/halluc_pilot_thinking/`) showed four of six still thinking at 4,096 tokens about a
  one-line question, at over a minute per answer. That would measure the thinking budget, not made-up answers, so
  the probe asks for the direct answer. (Of the two that finished, both said the thing does not exist.)
- Grading by pattern on the text after the thinking. Things that do not exist: "denies" (says it does not exist or
  that it does not know it) or "describes". Facts: right, wrong, abstains. Unfinished thinking is counted apart.
  The answers a pattern calls "describes" or "wrong" are read by hand for the cut models and for 9B.
- Reading (main comparison: cut model against 9B; 4B knows less, so it is the side reference): with 40 and 60
  questions and one sample, differences under 6 questions are not read. "More made up" = a cut model describes 6 or
  more of the 40 non-existent things than 9B does, or has 6 or more wrong facts than 9B, with the real-thing
  controls not lower. Otherwise: no difference shown at this size.

### Result (2026-11-13; Mac 4-bit, thinking off, one sample; `cache/halluc/`, graded by `halluc_probe.py grade`)

| model | 40 things that do not exist: described as real | 10 real things: denied | 60 facts: right | wrong | says it does not know |
|---|---|---|---|---|---|
| Qwen3.5-9B | 17 | 0 | 60 | 0 | 0 |
| Qwen3.5-4B | 21 | 1 | 57 | 3 | 0 |
| `g128b28160` + trained LoRA | **40** | 1 | **10** | 45 | 5 |
| `g128b28160` + old LoRA | **40** | 0 | **11** | 46 | 3 |
| 96×256 + old LoRA | **40** | 1 | **7** | 53 | 0 |

- By the rule fixed above this is "more made up", far past the 6-question line on both counts: the cut models
  describe every one of the 40 non-existent things as real (9B 17, 4B 21), and get 7–11 of 60 plain facts right
  where 9B gets 60 and 4B 57.
- What the wrong facts look like (read by hand, both `g128b28160` runs): wrong years (Battle of Hastings "1989",
  "1872"), wrong names ("John", "William", "Descartes" for Gauss), and **broken-off or mangled words where the right
  word was close**: "Titanium" for Titan, "Ttung" / "T Tung" for tungsten, "Farth" for farad, "Erit" for Asmara,
  "Kyrgyz" for Bishkek, "Menus" for Mars, "Mungol", "Kh", "Luk". The few right ones are the most common facts
  (72 for hafnium by chance or not, NaCl, 206, 1969).
- The real-thing controls only test denial; read by hand, the cut models' descriptions of real things are often
  wrong too (Bertrand's postulate "if two distinct lines intersect ... exactly one plane contains both").
- So in direct answers the cut model has lost most of its recall of names, years and numbers. This is the same
  damage as the name and word slips of 6h, seen without a thinking phase to work around it. It was invisible in
  the benchmark scores: multiple choice offers the names to pick from, math needs few rare words, and code fails on
  it (the check set's code gap).
- **Open, being run** (`cache/queue_o2.sh`): the same 60 facts and 10 real things with thinking on (4,096 tokens),
  for `g128b28160` + trained LoRA, 9B and 4B. The LoRAs were trained only on answers with thinking, so a model that
  recalls the facts with thinking on has the knowledge and has lost the direct-answer mode; a model that stays far
  below 9B has lost the knowledge. Reading fixed now: 50 or more of 60 right → the first; under 40 → the second.

### The same facts with thinking on (2026-11-13; 4,096 tokens; `cache/halluc_think/`)

| model | 60 facts: right | wrong | still thinking at the limit |
|---|---|---|---|
| Qwen3.5-9B | 59 | 0 | 1 |
| Qwen3.5-4B | 54 | 1 | 5 |
| `g128b28160` + trained LoRA | **12** | 41 | 7 |

- By the reading fixed above (under 40): **the knowledge is gone**, not only the direct-answer mode. Thinking does
  not bring the facts back ("Burkina" for its capital, "Mental Ocean" for the Mariana Trench, 1864 for Hastings).

## 6l. Was the knowledge cut by the choice of neurons? (design written 2026-11-13 09:30, before the run)

- Suspect: the neurons were chosen by their importance on 120 calibration windows of which 39 are code, 78 math and
  3 anything else (6h). Nothing in them asks the model to recall who, where or when. Neurons that only work on such
  text score zero and are the first to go. The check sets (multiple choice, math, code) could not show the loss.
- General windows: no such text is on the Mac and nothing is downloaded for this. Qwen3.5-9B (59–60 of 60 on the
  facts) writes them: 300 topics from a list written for this (history, places, science, arts, technology, people,
  daily life; `heal/general_text.py`), three kinds of request (an overview with names and dates, a list of key
  facts, quiz questions with answers), thinking off, cut to 512 tokens in the chat format. The topics are
  broad (capitals, elements, moons, dates), so the generated text will name some of the probe's answers; this is a
  choice of neurons, not training, but the probe is a development measure from here on. Two guards: a second set of
  60 facts (`halluc_probe.py`, FACTS2) is written now and asked only of the final choice, and both sets are also
  reported split by whether the answer appears anywhere in the general windows.
  The text needs not be right in every detail: it is fed to the 35B model, and what is measured is which of its
  neurons such text uses.
- Pass (`moe/calib_pass.py`, Mac MPS float32, one pass over the original): statistics on the chat windows (as
  before) and on 150 general windows; shape `g128b28160` chosen from (a) chat statistics = the current model,
  (b) the two statistics each scaled to sum 1 per layer and added, (c) general statistics only. KL to the original
  model's own output on 60 held-out chat windows and on 40 held-out general windows.
- Added 09:45, still before the pass: a second mix with the general statistics at a quarter weight is scored too.
  It is judged by the same rule only if the half-and-half mix fails the chat limit. (A one-layer dry run on 15
  windows, CPU, to test the program: the current choice keeps 61% of the chat importance and 43% of the general
  importance of layer 1.)
- Diagnostic reported first: the share of the general windows' importance that sits in neurons the current model
  kept, against the same share for the chat windows.
- Rule: build (b) for the Mac if its KL on the general windows is at least 20% under (a)'s while its KL on the chat
  windows is at most 0.01 above (a)'s. Then, with the old LoRA (it loads on any choice of neurons): the 60 facts and
  the 40 non-existent things, the agent tasks, and the check set with the 8,192 cap. It replaces the current model
  if the facts go up by 15 or more and the check set stays within 3 of 43; the second fact set and one unseen
  benchmark set are then run once. If (b) fails the KL rule, the loss is not a matter of which neurons were kept at
  this size, and that is the result.

### Result of the pass (2026-11-13; Mac MPS float32, 1.6 h; `cache/general/calib_results.json`)

217 general windows were used for the statistics (the design said 150; the text gave 257 windows, 40 held out).

| neurons chosen from | KL, chat windows | KL, general windows | neurons shared with the current model |
|---|---:|---:|---:|
| (a) chat windows = the current model | 0.150 | **2.365** | — |
| mix, general at a quarter weight | 0.170 | 1.284 | 82% |
| (b) mix, half and half | 0.223 | 0.915 | 64% |
| (c) general windows only | 0.966 | 0.567 | 35% |

- Diagnostic: the neurons the current model kept hold 61% of the importance measured on chat windows and 29% of the
  importance measured on general windows (per layer 44–70% and 19–46%; lowest in layers 15–23). The experts it kept
  receive 75% of the router weight on chat text and 48% on general text.
- The current model is 16 times further from the original on general text than on the text it was chosen on. The
  check sets and every KL in this file so far used the chat windows and could not see it. This matches the probe (6j).
- **By the rule as written neither mix is built**: both cut the general KL by more than 20% (61% and 46%) but both
  raise the chat KL by more than 0.01 (0.073 and 0.020).
- The rule's closing sentence does not follow from these numbers, though. Which neurons are kept moves the general KL
  by a factor of four; what fails is having both at this size. Even the choice made for general text alone stays at
  0.567 there, four times what the chat choice reaches on chat text: general text leans on more neurons than code and
  math do, so 9.4B cannot hold both fully.
- The 0.01 limit was mine and was set from a one-layer dry run where KL was 0.03. On the whole model it is 7% of the
  chat KL and half of the 0.02 gap between the float32 pass and the Mac 4-bit build.

### An exploratory build after the failed rule (written 2026-11-13 11:55, before the build)

- The rule failed, so nothing below can replace the current model on this evidence alone. The Mac is otherwise idle,
  and how many facts come back is what the next training needs to know, so two builds are made anyway:
  the quarter-weight mix (chat KL 0.170, still under every 8.5B shape of 6h) and the general-only choice as an upper
  bound for what the choice of neurons can return.
- Measured with the old LoRA, as planned: both on the probe (60 facts, 40 non-existent things); the quarter mix also
  on the agent tasks and the check set with the 8,192 cap.
- Reading, fixed now: the quarter mix is a candidate only if the facts go up by 15 or more and the check set stays
  within 3 of 43 (the old replacement rule); a candidate is then confirmed once on the second fact set and one unseen
  benchmark set before it replaces anything. If the general-only build stays under 30 of 60 facts, the choice of
  neurons cannot return the knowledge at this size and it has to come from training or from more parameters.

### The exploratory builds on the probe (2026-11-13; Mac 4-bit, old LoRA, thinking off, one sample; `cache/halluc/`)

"Named" = the right answer stands in one of the 294 generated general texts next to a rare word of the question
(`halluc_probe.py grade --general`): 37 of the 60 facts. 9B has 37/37 and 23/23, 4B 35/37 and 22/23.

| neurons chosen from | facts right (60) | named (37) | not named (23) | wrong | says it does not know | fake things described (40) |
|---|---:|---:|---:|---:|---:|---:|
| chat windows = the current model | 11 | 3 | 8 | 46 | 3 | 40 |
| mix, general at a quarter weight | **32** | 22 | 10 | 28 | 0 | 40 |
| general windows only | **43** | 30 | 13 | 13 | 4 | 39 |

- The facts come back with the choice of neurons alone, no training: 11 → 32 with a quarter weight, 43 with the
  general-only choice (over the 30 set as the line above). The original model's knowledge is still in the weights
  the cut keeps or drops; the first choice dropped it because nothing in its calibration text asked for it.
- Most of the gain is in facts the calibration text names (3 → 22 → 30 of 37); the others move less
  (8 → 10 → 13 of 23). The choice returns what the text covers, so the breadth of that text decides what is kept, and
  the 60 facts overstate the gain for facts in general. The second fact set (36 of 60 named) will show the same split.
- Describing things that do not exist does not change (40, 40, 39; the grader's four "denies" for the mix and one of two for the general-only
  choice were descriptions that tripped its patterns, read by hand and counted as descriptions here). Knowing more facts did not bring back
  "I can't find that".
- Many of the mix's wrong answers are near misses in the spelling: "Astan", "Mans", "Ouumb", "Brgyzstan", "Farth",
  and two right ones carry a doubled first piece ("T Triton", "M Mariana Trench") — the slip of 6h's word count.

### The quarter mix on the check set and the agent tasks (2026-11-13; old LoRA; `cache/heal_grades/mix25_a/`)

| neurons chosen from | no cap (16k) | thinking cap 8,192 | closes its thinking by itself | agent tasks (48) | wrong e-mail or id |
|---|---:|---:|---:|---:|---:|
| chat windows = the current model | 43 (14 / 18 / 11) | 43 (13 / 19 / 11) | 53 of 60 | 37 (42) | 4 episodes |
| mix, general at a quarter weight | 34 (10 / 14 / 10) | **42** (12 / 20 / 10) | 40 of 60 | 40 (42) | 1 episode |

- By the reading fixed before the build the mix is a **candidate**: facts +21 (15 needed), the capped check set 42
  against 43 (within 3), code 10 against 11 (the plan's guard is a loss of 2). No made-up name in its 14 finished
  code answers (`name_slips.py`; the current model has 1 in 19); private unknown words in prose 39.4 per 10k against
  39.0, no change.
- What it costs: the thinking closes less often (40 of 60 against 53; 13 answers run into loops), so without the
  cap it loses 9 questions. With the cap math is 20 of 20. The mix leans on the cap more than the current model does;
  the same was seen after the Kaggle session (41 of 60), so this LoRA-and-shape pair sits close to the edge either way.
- Agent tasks: 40 solved against 37 (42 and 42 with an answer left in the thinking counted), wrong e-mail or order id
  in 1 episode against 4, but calls with bad arguments in 16 episodes against 4:
  `write_file` sent without its content and `update_order_status` with `path` for the order id; two episodes repeat
  such a call until the turn limit.
- Confirmation, as fixed: the second fact set (also asked of the current model and 9B, to have something to compare
  with) and the unseen `check_set2` with the cap (the current model: 44 without, 46 with). It replaces the current
  model if the second fact set is up by 10 or more of 60 and `check_set2` with the cap is within 3 of 46. (The 10 is
  set now, before the run: 36 of the 60 second facts are named in the general text, about the same as the first set.)
  The tool `think_cap.py` stopped on an answer that had closed its thinking at once; such answers now pass unchanged.

### Confirmation 1: the second fact set (2026-11-13; 60 facts never asked before, thinking off; `cache/halluc2/`)

| model | right (60) | named in the general text (36) | not named (24) | wrong | says it does not know |
|---|---:|---:|---:|---:|---:|
| current cut + old LoRA | 15 | 11 | 4 | 37 | 8 |
| quarter mix + old LoRA | **36** | 22 | 14 | 23 | 1 |
| Qwen3.5-9B | 59 | 36 | 23 | 1 | 0 |

- Up by 21, the same as on the first set (10 were needed). Here the facts the calibration text does not name rise as
  much as the named ones (4 → 14 of 24, 11 → 22 of 36), so the gain is not only the text's own facts.
- 9B's one miss is "PLN" for the currency of Poland, which the pattern (złoty) does not accept.
- Still to come: `check_set2` with the cap.

### Confirmation 2: `check_set2` (2026-11-13 20:30; old LoRA; `cache/heal_grades/mix25_b/`)

| model | no cap (sci / math / code) | cap 8,192 | answers that needed the cap |
|---|---:|---:|---:|
| current cut + old LoRA | 44 (14 / 13 / 17) | **46** (14 / 15 / 17) | 15 |
| quarter mix + old LoRA | 39 (15 / 13 / 11) | **41** (15 / 14 / 12) | 28 |

- **The rule fails: 41 is 5 under 46, and the loss is in code (17 → 12).** The quarter mix does not replace the
  current model. Question by question with the cap: science won 1 lost 0, math won 2 lost 3, code won 1 lost 6. Five
  of the six lost code questions are answers that looped in the thinking and needed the cap. No made-up name in
  either model's finished code (`name_slips.py`: 0 and 0), so the cost is loops, not invented functions.
- So at this size the facts come back by the choice of neurons (11 → 32, 15 → 36 of 60) but code on an unseen set
  pays for it. The check set (dev) had not shown it (code 10 against 11): one more case where the unseen set decides.
- A problem on the way: the capped run was stopped eight times in seven minutes by the memory guard (free memory
  under 10% with the server at 14 GB while the Mac was in use). Fixed: `heal/mem_wait.sh` waits for free memory before
  a server starts (`mac_eval.sh`, `mac_think_cap.sh`), and the capped run was redone with one request at a time.

## 6m. Notes in the prompt, and does the model know when it is unsure? (design written 2026-11-13 16:25, before any run)

Two questions behind the plan for the next step (retrieval for the missing knowledge; a little training at most).
`heal/rag_probe.py`; the 60 facts of 6j; thinking off, the usual sampler, one sample unless said.

- **Notes.** Each fact is asked twice with eight one-line notes above the question and "using only the notes …
  if the notes do not contain the answer, say 'not in the notes'": once with the note that states the answer among
  seven notes for other facts, once with eight notes for other facts. Counted: with the note, right or not (a
  wrong spelling of a name that is in the notes counts as wrong); without it, says so / right from memory / made up.
  Models: the current cut, the quarter mix (both with the old LoRA), 9B, 4B.
  Reading: retrieval can stand in for the lost knowledge if the cut model is right on 54 or more of 60 with the note.
  If it says "not in the notes" on fewer than 30 of 60 without it, examples of exactly that go into the training.
- **Unsure or not.** The 60 facts without notes, five samples each, for the two cut models. A question's answer is
  the most frequent one; its agreement is how many of the five give it. Counted: how often the most frequent answer
  is right at agreement 5, 4, 3 and below. Reading: the model can tell if answering only at agreement 4 or 5 at
  least halves the share of wrong answers while keeping two thirds of the right ones. Then "I don't know" can be
  taught with few examples or added outside the model; if not, it cannot be taught cheaply and only more knowledge
  helps. The 40 non-existent things are not covered: their answers are free text and five of them cannot be compared
  by a program.

### Result (2026-11-13; Mac 4-bit, thinking off, about a minute a run; `cache/open/`, `cache/rep/`)

| model | with the note: right (60) | without it: says "not in the notes" | right from memory | made up |
|---|---:|---:|---:|---:|
| current cut + old LoRA | 60 | 56 | 1 | 3 |
| quarter mix + old LoRA | 60 | 60 | 0 | 0 |
| Qwen3.5-9B | 60 | 60 | 0 | 0 |
| Qwen3.5-4B | 60 | 59 | 0 | 1 |

| five samples, no notes | most frequent answer right / wrong, all 60 | only where 4 or 5 samples agree | right answers kept |
|---|---:|---:|---:|
| current cut + old LoRA | 12 / 40 (77% wrong; 8 "I don't know") | 11 / 6 (35% wrong) | 92% |
| quarter mix + old LoRA | 33 / 27 (45% wrong) | 31 / 5 (14% wrong) | 94% |

- **Notes work.** With the answer eight lines above the question the cut models are right on all 60, every name
  spelled as in the note, and without it they say so (56 and 60). By the reading, retrieval can stand in for the lost
  knowledge, and no "not in the notes" examples are needed in the training. The limit of this test: eight short lines
  and one sample. The agent tasks (6k), with longer contexts and thinking on, did show miscopied e-mail addresses.
- **The model can tell.** When it is wrong its samples scatter (wrong answers at agreement 1–2: 29 of 40 and 19 of
  27); when all five agree it is mostly right (11 of 13 and 27 of 32). Answering only at agreement 4 or 5 more than
  halves the wrong share for both models and keeps over 90% of the right answers. So the uncertainty is there to be
  used: by a few "I don't know" examples in the SFT, or outside the model by asking several times.
- What is left after that filter is the confidently wrong answer: 6 and 5 questions.

## 6n. More knowledge for the same cost on chat text? Three ways to choose (design written 2026-11-13 16:40, before the run)

The user's question: can the knowledge come back without paying on code, math and science. At a fixed size every
knowledge neuron takes a seat, so the question is whether a better way of choosing gets more general KL back per
unit of chat KL than the plain mix does. One more pass (`moe/calib_pass2.py`, Mac MPS, about 2 h), same shape and
total size, same held-out windows as 6l:

- `mix12`: the plain mix with the general statistics at one eighth. It gives the plain mix a fourth point:
  (chat KL, general KL) = (0.150, 2.365) at 0, (0.170, 1.284) at a quarter, (0.223, 0.915) at a half.
- `mix25x`: the quarter mix with the seats moved **between layers**. Every layer has had 28,160 neurons. Here one
  threshold on a neuron's share of its layer's (mixed) importance is used for all layers, total unchanged, each
  layer held between 0.6 and 1.4 times 28,160. 6l found the middle layers (15–23) keeping the least of the general
  importance.
- `mix25p`: the quarter mix where a neuron's general importance is its mean over the **22 windows (10%) that use it
  most**, not over all 217. A neuron that matters only for one topic is used rarely and averages low.
- `mix25q`: the quarter mix where the general importance is the **squared** activity summed over tokens (the energy
  the output loses when the neuron goes) instead of the absolute activity. The chat statistics stay as stored.
- Reading: the plain mix's four points joined by straight lines are the line to beat. A way of choosing is better if
  its general KL is at least 10% under that line at its own chat KL. Only then is it built and asked the facts and
  the check set; otherwise the plain quarter mix stays the candidate and the answer to the question is "not by
  the way of choosing, at this size".
- Added 2026-11-13 20:35, while the pass was at layer 3 of 40 and before any of its numbers: the plain quarter mix
  has since failed on `check_set2` (code 17 → 12, loops), at a chat KL of 0.170 against 0.150. A variant at the same
  chat KL would pay the same. So a variant, `mix12` included, is built only if its chat KL is at most 0.160 and its
  general KL at most 1.80 (half of the quarter mix's gain); it is then asked the facts and the check set under the
  code guard of section 7. `check_set2` has now decided against a candidate once, so a later candidate needs a set
  not used so far for its confirmation.

## 6o. Perplexity, next to KL (2026-11-13 21:25; the author's request; `moe/mlx_ppl.py`, `cache/ppl.json`)

Perplexity on the stored 512-token windows (exp of the mean next-token cross-entropy). Mac 4-bit builds; the
original's value is the one stored when the windows were made (bf16). Wikipedia is human-written English (60
windows), chat is the original's own thinking answers to code, math and science questions (60 held-out windows),
general is the general-knowledge text of 6l (40 held-out windows; written by 9B, so 9B's own value there flatters it).

| model | Wikipedia | chat (code / math / science) | general (written by 9B) |
|---|---:|---:|---:|
| original 35B-A3B | 7.64 | 1.60 | |
| Qwen3.5-9B | 9.41 | 1.79 | 1.90 |
| Qwen3.5-4B | 11.42 | 1.86 | 2.81 |
| current cut + old LoRA | **89.6** | **1.86** | 11.0 |
| current cut, no LoRA | 171.2 | 1.89 | 21.9 |
| quarter mix + old LoRA | 90.1 | 1.88 | 6.05 |

- **On the kind of text it was cut and measured on, the cut model is at 4B's level and near 9B (1.86 against 1.86
  and 1.79). On Wikipedia it is ten times worse than 9B (90 against 9.4).** The KL on chat windows, the only ruler
  since rehearsal 4, could not show this; the benchmark sets are all of the chat kind too. It is the same finding as
  the lost facts (6l), now as one number.
- The old LoRA halves the Wikipedia perplexity (171 → 90) though it was trained on chat answers and UltraChat only.
  The quarter mix halves the general text's (11 → 6) and leaves Wikipedia where it was.
- What it means for "beating 9B": level on benchmark questions, far behind as a general model. A fair comparison
  with 9B has to include text like this from now on; `mlx_ppl.py` takes a minute a model.

## 7. Plan agreed with the author (2026-11-13 16:30)

Aim: fewer made-up answers with as little training as possible, and no loss on code. Three parts: the knowledge comes
back by the choice of neurons (6l), copying slips and made-up answers are met with a little SFT, retrieval is an
addition for use and not the main road (benchmark science and API names are recalled without it).

- **Guard on code, at every step.** A candidate is dropped if the check set's 20 code questions lose 2 or more against
  the current model (11 without a cap, 11 with it, old LoRA) or if the count of made-up names in its code answers
  (`heal/name_slips.py`, static) goes up. A third or more of any SFT data is the teacher's code answers (265 exist);
  no "I don't know" example is about code.
- **Stage 1, Mac, no training (today).** The quarter mix on the check set (within 3 of 43, 6l); notes in the prompt
  and agreement between samples (6m).
- **Stage 2, the base.** The current choice or a mix, by stage 1; a weight under a quarter and a wider calibration
  text (1.6 h on the Mac) are open if the quarter mix costs too much on the check set.
- **Stage 3, a little SFT (Kaggle T4×2 after the quota returns, about an hour).** First the teacher itself on the
  probe (10–15 min of GPU): how many facts the original knows and how many non-existent things it describes is the
  ceiling for everything here. Then `moe/bench_uneven.py --device cuda --dtype float16`. Data by stage 1: answers
  from notes and "not in the notes" if copying fails; "I don't know" examples only if the model can tell when it is
  unsure; code answers mixed in; LoRA on the expert side too.
- **Stage 4, confirmation.** The second fact set (unused), the agent tasks, the check set; then one unseen benchmark
  set once; the exact benchmark on T4×2 if it holds.

**Colab units (2026-11-13 19:15; balance 103.46 CU, 200 more on 11-23).** The user wants them used but not wasted.
What the Mac and the free Kaggle T4×2 can do stays there (selection passes, the 60-question sets, the small SFT). Colab
is for what needs the original model in bf16 or a large GPU (G4, 96 GB, about 10 CU an hour by the 11-08 run):

1. The original 35B-A3B on the probes (`heal/colab_teacher.sh`, vLLM, not yet run): 60 facts + 40 non-existent
   things, the second 60 facts, the notes and agreement tests, the check set and `check_set2`. About 45 minutes,
   about 8 CU. It answers whether the original knows less than 9B, whether it also describes things that do not
   exist, and how far the cut model is from it on the unseen set. Nothing is tuned on it.
2. One longer recovery run, only if the base is settled and (1) shows room: 30 CU at most.
3. The exact benchmark at the end, the final model against 9B, once: about 20 CU.

The rest (about 45 CU) is kept until 11-23.

### The original model on the probes (2026-11-13 19:49–20:20; Colab G4, vLLM 0.33, bf16; `cache/teacher/`)

31 minutes of VM time, 4.92 CU (balance 98.54): 14 minutes to install and load, 1 minute for the 590 short answers,
15 minutes for the two check sets. Same prompts and sampling as on the Mac, no seed. The original runs in bf16, the
others below are 4-bit Mac builds.

| model | facts (60) | second facts (60) | non-existent things described (40) | of the 11 about code |
|---|---:|---:|---:|---:|
| original 35B-A3B | 59 | 60 | **8** | 0 |
| Qwen3.5-9B | 60 | 59 | 17 | |
| Qwen3.5-4B | 57 | | 21 | |
| quarter mix + old LoRA | 32 | 36 | 40 | |
| current cut + old LoRA | 11 | 15 | 40 | 11 |

| model | check set (sci / math / code) | `check_set2` |
|---|---:|---:|
| original 35B-A3B, no cap | **51** (16 / 18 / 17) | **51** (18 / 14 / 19) |
| Qwen3.5-9B, cap 8,192 | 46 (15 / 16 / 15) | 45 (16 / 12 / 17) |
| current cut + old LoRA, cap 8,192 | 43 (13 / 19 / 11) | 46 (14 / 15 / 17) |

- **The original does not know less than 9B** (119 against 119 of 120). Its one miss is "the Challenger Deep" for
  the deepest trench. With the notes: 60 of 60, and "not in the notes" 60 of 60 without. Five samples: all five
  right on all 60.
- **Making things up comes from the cut, not from the original.** The original says "this does not exist" for 32 of
  the 40, more often than 9B (23), and for all 11 that ask about a function, a library or a command. The 8 it
  describes are named theorems, effects and one film. So examples of saying "this does not exist" do not go against
  the original: its own answers can be the SFT data.
- **The ceiling on the unseen set is 51, the cut model is at 46, 9B at 45.** By field on `check_set2` the original
  leads the cut model in science (18 against 14) and code (19 against 17) and not in math (14 against 15; it ran
  out of the 16,000 tokens on 6 math questions). Three of the 6 math questions the original missed the cut model
  got right, so single questions move by chance at this size of set.
- What it changes in the plan: stage 2 keeps the current choice of neurons (the quarter mix is out, see 6l
  confirmation 2) unless 6n finds a cheaper one. Stage 3's data is the original's answers: to questions about things
  that do not exist (new ones, none from the probe; code ones included, since the original refuses all of them),
  to short facts, and the code answers already made. Facts the cut has lost do not come back from a little SFT;
  for those the notes test stands (60 of 60 with the note).

## 8. A small SFT against made-up answers (design written 2026-11-13 21:00, before any run; `heal/honesty_data.py`)

The user's request: train, on Colab, a model that makes things up less. What 7 found sets the shape of it: the
original says "this does not exist" where the cut model describes, so the lesson can be the original's own answers.

- **Base**: the current model (`g128b28160`, chat choice of neurons), LoRA started from the 6M LoRA, the same targets
  as before (attention, GDN, shared expert; rank 16). No LoRA on the routed experts: neither `ExpertLoRA` nor the
  Mac adapter handles experts of different widths yet, and writing both under a running VM is how units get wasted.
- **Data**, all answers by the original 35B-A3B (vLLM, bf16; half with thinking on, half off):
  - *Things that do not exist*: 396 Python names made by recombining real names of the same module (`hasattr` says
    they do not exist; `itertools.takewhile_safe`, `json.safe_loads`) and 420 named results from invented surnames
    ("the Sandvold–Oszlanyi test"). Kept only when the original says it does not exist.
  - *Real things in the same forms*: 341 real names of the same modules and 154 real named results (two forms each),
    kept only when the original describes them. Without these the lesson would be "deny everything".
  - *Facts*: the original writes about 1,500 quiz questions with short answers over 50 topics; a fact is kept when
    the original gives its own answer again three times out of three. The cut model (Mac, old LoRA) answers each five
    times: "known" at four or five right, "not known" at none or one. Known facts are trained with the original's
    answer, not-known ones with a short "I don't know" (written here, 3–4 wordings), the same number of each, 300 at
    most. A fact whose answer matches any pattern of the two fact probes is left out.
  - *The guard on code*: all 261 code answers of the original, math and science answers up to the same number of
    tokens, and the honesty examples cut back if they would outweigh the code answers. UltraChat stays at 30%.
- **What is not in the training data**: the probe's 40 non-existent things, its 120 facts, its question forms
  ("What does the Python standard library function … do?"), the invitation to say "I don't know", and the forms and
  names of the transfer probe below.
- **Transfer probe** (`fix_task/halluc_set3.jsonl`, 45 questions, written before the run): forms the SFT never sees.
  Ten "write a function … use `<name that does not exist>`", ten "difference between `<real>` and `<not real>`", ten
  "who first established the `<invented result>`, and in which year", and fifteen with real things in the same forms.
  Graded by `honesty_data.py grade3`: says it does not exist / goes along, and for the code tasks whether the
  finished code calls the name. Asked of the original, 9B, 4B and the cut model before training.
- **Run**: one Colab G4 session. (A) the original answers, about 10 minutes after loading; (B) the student is cut on
  the VM, about 15 minutes, while the Mac marks the facts; (C) a three-step check, then one pass over the data, 60
  minutes at most, LoRA saved at 0.5M and 1M tokens and at the end. Expected 1.5 to 2 hours, 15 to 20 CU (the plan
  of 7 allows 30).
- **Reading, fixed now.** The LoRA from the end of the run is the one that is judged; an earlier one is looked at only
  if the last fails the guard, and that is then said. All on the Mac, 4-bit, against the current model + old LoRA:
  1. Guard: check set with the 8,192 cap, code not down by 2 or more (11), total not down by 3 or more (43), science
     not down by 2 or more (13; a model that says "I don't know" to benchmark questions would show here); no more
     made-up names in finished code (`name_slips.py`).
  2. It works if, on the transfer probe, "goes along" falls by a third or more and no more than one real thing more
     is denied; the probe's 40 (now 40 described) and its real 10 are reported next to it as the near form.
  3. Facts (both probes): right answers not down by more than 2 of 60, and "wrong of answered" down.
  If 1 holds and 2 or 3 holds, it goes on to a set not used so far for the totals; if 1 fails, it is not adopted.

### Run 1 (2026-11-13 21:49–23:35; Colab G4; `cache/honesty/run1/`, adapter `cache/heal_adapters/honest1`)

- Data: the original answered 3,978 + 1,326 fact questions, 1,465 thing questions and the 45 of the transfer probe in
  about 23 minutes. 679 thing answers were cut off by the token limits (512 with thinking off, 3,000 with it on;
  the original thinks long about whether a named result exists) and were dropped. Of 1,326 written facts, 123 are
  known to the cut model (4–5 of 5 right), 869 not known (0–1 of 5). The mix: 948 examples, 1.62M tokens fed, answer
  tokens a third each for honesty, code and math with science. One pass, 44 steps, 21 minutes; KL A+B on chat
  windows 0.1464 → 0.1336 (it went down, not up).
- Export for vLLM (`moe/pad_uneven.py`): the uneven model written as an ordinary 128-expert, width-512 checkpoint
  (zero-padded; dummy experts get a router row that is never picked). On 105 held-out sequences no dummy was picked,
  the top token agrees on 99.5% of positions (bf16 against fp16 paths), KL to the original 0.1336 / 0.1335.
- Transfer probe (thinking off, Mac 4-bit):

| model | non-existent (30): goes along | of 10 code tasks: calls the name | real (15): denies |
|---|---:|---:|---:|
| original | 8 | 2 | 1 |
| before (current + old LoRA) | 25 | 7 | 1 |
| **after run 1** | **18** | **7** | 1 |
| 9B | 14 | 3 | 0 |

- The probe of 6j (near form): non-existent things described 40 → 33, real things denied 0 → 2. Facts: first set
  right 11 → 11, wrong 46 → 31, "I don't know" 3 → 18 (wrong of answered 0.81 → 0.74); second set right 15 → 13,
  wrong 37 → 32, "I don't know" 8 → 15 (0.71 → 0.71).
- Guard (Colab G4, vLLM bf16, no cap, one sample; check set 60; `cache/guard/`):

| model | science | math | code | total | thinking closed |
|---|---:|---:|---:|---:|---:|
| before (current + old LoRA) | 12 | 17 | 11 | 40 | 54 |
| after run 1 | 10 (lenient 14) | 15 (16) | **14** (15) | 39 | 45 |

  Code is up by 3, the total down by 1, **science down by 2 (strict)**: the trained model leaves its thinking open
  more often (science: closed 18 → 12, loops 0 → 5). So run 1 also fails the guard on science.
- **Reading by the rule: 2 fails** (25 → 18 is a fall of 28%, a third needs 16 or fewer). 3 holds on the first set
  and on the two together (0.76 → 0.72), not on the second alone. The guard (1) is measured on Colab bf16 without
  the cap for both models (the Mac with the cap would take three hours a model): a change of the plan, written here.
  Not adopted as it is. The code tasks did not move at all: no training question asked for code.

### Run 2 (design written 2026-11-13 23:40, before the run; the author: "train first, the benchmark can wait for T4×2")

- Same base, same LoRA start (the 6M LoRA, not run 1's), same targets and rate.
- Data, all from the original again:
  - The 679 thing questions whose answer was cut off, asked again with 1,500 tokens (thinking off) and 8,000 (on).
  - A code-usage form, new: "Show a short Python example that calls `x`", "Write a few lines of Python that use
    `x`", "Give me a code snippet using `x`" for 264 new non-existent names and 90 new real ones (`make2`; none of
    the transfer probe's or the probes' names). This brings training nearer to the transfer probe's code tasks
    ("write a function that …; use `x` for it"); its other two forms (difference, who and when) stay unseen.
  - Facts and the rest as in run 1. Shares as in run 1 (code answers at least a third).
- Two passes over the mix (every example twice, each pass shuffled), about 90 steps.
- Reading: the same three rules as run 1, against the same "before" (the guard on Colab bf16 without the cap, as for
  run 1). Thinking closed on the check set is reported next to the guard: run 1 lost 9 of 54. The code tasks of the transfer probe are
  reported, with the note that they are now near the training form.

### Run 2 result (2026-11-14 00:51–01:41; `cache/honesty/run2/`, adapter `cache/heal_adapters/honest2`)

- The first start killed itself (`pkill -f "vllm serve"` in the job's own command line matched the job); 40 minutes of
  G4 were lost before anyone looked. Fixed in the scripts (`pkill -f "bin/vllm serve"`).
- Training: 3.22M tokens fed, 85 steps, 35 minutes, KL A+B 0.1464 → 0.1357. Export: no dummy picked, KL 0.1357 /
  0.1357.

| model | transfer: goes along (30) | calls the name (10) | real denied (15) | 6j: fakes described (40) | facts 1: right / wrong / idk | facts 2 |
|---|---:|---:|---:|---:|---|---|
| before | 25 | 7 | 1 | 40 | 11 / 46 / 3 | 15 / 37 / 8 |
| run 1 | 18 | 7 | 1 | 33 | 11 / 31 / 18 | 13 / 32 / 15 |
| **run 2** | **18** | **4** | **0** | **21** | 12 / 38 / 10 | 14 / 36 / 10 |
| original | 8 | 2 | 1 | 8 | 59 / 1 / 0 | 60 / 0 / 0 |
| 9B | 14 | 3 | 0 | 17 | 60 / 0 / 0 | 59 / 1 / 0 |

| guard (check set, Colab bf16, no cap) | science | math | code | total | thinking closed |
|---|---:|---:|---:|---:|---:|
| before | 12 | 17 | 11 | 40 | 54 |
| run 1 | 10 | 15 | 14 | 39 | 45 |
| **run 2** | **13** | 15 | **12** | **40** | 49 |

- **Reading by the rule.** 1 (guard) holds: code +1, total equal, science +1. 2 fails again: 18 of 30 go along, 16
  were needed; what moved are the code tasks (7 → 4; their form is now near the training form, so this is the
  weaker kind of evidence) and the near form of 6j (40 → 21 described). 3: right answers 11 → 12 and 15 → 14; wrong of
  answered 0.81 → 0.76 on the first set, 0.71 → 0.72 on the second, 0.76 → 0.74 on the two together. Read strictly
  (each set), 3 fails on the second set by 0.01. **So by the rule written before the runs, run 2 is not adopted
  either**; it is better than "before" on nearly every line, which is for the author to weigh, not for the rule.
- The forms that did not move are the unseen ones: "difference between `real` and `not real`" and "who first
  established X, and in which year". The model learned "this name does not exist" for the forms it saw, little more.
- Colab: about 38 CU for the evening (103.46 → 60.18 after the probe run of 7), of which about 6.5 lost to the
  self-killed start.

## 9. The exact benchmark on 2x T4 (design written 2026-11-14 01:45, before the run; `heal/kaggle/bench_uneven_{a,b}`)

The goal of 11-12: 46 of 60 on the usual 60, then the exact benchmark on T4×2 (no G4 for it). The Mac reached it
with the 8,192 cap (usual 60: 50 with either LoRA; `check_set2`: 46). What is left is the exact run.

- **How the uneven model runs on vLLM**: `moe/pad_uneven.py` folds the LoRA and writes an ordinary Qwen3.5-MoE
  checkpoint (128 experts of width 512 in every layer, zero padding; dummy experts whose router row is never in the
  top 8). In float16 it is about 35 GB, more than two T4s hold, so vLLM keeps 7 GB per GPU in CPU memory
  (`--cpu-offload-gb 7`): slower, same numbers. Before serving, the kernel counts on the uneven model itself whether a
  dummy would reach the top 8 on held-out windows and on the 60 benchmark prompts; one hit stops the run.
- **Settings as for the references**: vLLM 0.33, float16, tensor parallel 2, thinking on, 16,000 tokens, temperature
  0.6, top-p 0.95, top-k 20, seed 0; then the same answers under the 8,192 thinking cap (`think_cap.py`). Code is
  run on the kernel's VM.
- **Models**: `a` = the current cut + old 6M LoRA (the model that reached 46 on the Mac); `b` = run 2 of section 8
  (not adopted by its rule; benchmarked so that the author can weigh it). Two kernels at once (Kaggle allows two GPU
  sessions). 9B on the same settings is a later run if the quota allows.
- **Reading**: the 60 are the development set of this project, so this is the "exact" number for the record, not a
  new confirmation. The capped score is compared with the Mac's 50 (same model, 4-bit): a difference of more than 3
  is a sign that 4-bit or the export changed the model, and is looked into before anything is published.
- Start: pushed by a waiting job on the Mac at 09:05 (the quota is believed to return at 09:00), retried every 15
  minutes until noon; the kernel log is read 10 minutes after the start.
- **09:55, first push stalled.** Both kernels read the student straight from `/kaggle/input`: the load report came
  after 570 s (the training kernel, which copies to `/tmp` first, had it after about 90 s), then nothing for over 40
  minutes, still inside the load of the routed experts (54 s from `/tmp` in the training kernel). The kernels now copy
  the student to `/tmp` first, as the training kernel does. The Kaggle CLI cannot cancel a run and the API's cancel
  call needs a session id it never returns, so the old runs have to be cancelled on the website;
  `cache/repush_bench.sh` pushes the fixed kernels as soon as the old runs end in error or are cancelled.

- **Result (11-15 01:20): neither run finished.** Both ran until Kaggle's 12-hour limit and were cancelled at 01:17:
  run a (current model, 6M LoRA) answered 54 of the usual 60 (4 math and 2 science missing), run b (honest2) 42
  (14 science, 3 math, 1 code missing). No capped score, nothing on the other sets.
- **Why so slow:** generation ran at 3-7 tokens/s. The padded checkpoint is ~35 GB in float16 (T4 has no bfloat16;
  vLLM cast it), more than the 2x15 GB, so vLLM put 7.2 GB per GPU in CPU memory (`Total CPU offloaded parameters:
  7.23`) and read it over PCIe at every step; the KV cache held 6 answers of 20k tokens at a time. Padding every
  expert to 512 is what makes it too big: the uneven model itself is ~19 GB.
- **What the 54 answers say (graded on Kaggle, loose, `cache/kaggle_bench_out/`)**: on the same 54 questions the
  float16 model has **50** (science 15/18, math 16/16, code 19/20) and the Mac 4-bit answers of the same model (no
  cap) 43 (15 / 15 / 13); 40 right in both, 10 only on T4, 3 only on the Mac (exact sign test p = 0.09). None of the
  T4 answers went past 16k tokens, so the Mac's lower token limit does not explain it; 5 of the 10 were Mac answers
  that looped to the limit, 5 were short code answers that were simply wrong on the Mac. One sample each, and the
  6 unanswered questions are the longest ones, so 50 of 54 is not a score of 60; but the direction is that the Mac
  4-bit numbers understate the model (code and loops most), not the reverse. Run b: 41 of its 42 (science 6/6, math
  17/17, code 18/19), with the hard science questions the ones missing.
- Not yet measured exactly. The model card says so. A run that fits needs a smaller form for T4 (or a larger GPU);
  the Mac's disk is at 99% (13 GB free), so an 8-bit Mac build is not started without the author.

### 9b. The same benchmark on the G4 (design written 2026-11-15 01:35, before the run; the author: "after wikil-a, in the
same session; if it fails, give up")

- The current model (6M LoRA) only, bf16, in the run 3 session after wikil-a, if 6+ units are left then
  (`heal/colab_bench.sh`, driven by `cache/colab_run3_go3.sh`). The steps of the T4 kernel: the LoRA folded into the
  padded checkpoint (`pad_uneven.py`; stops if any padding expert would be picked), vLLM, seed 0, 16k tokens, then the
  8,192 cap of the same answers; then the second fresh set and the check set the same way, as far as the units go
  (cut off below 1.5 units or after 70 minutes; finished stages are kept). Graded on the VM (code executed there).
- Read against the Mac 4-bit numbers of the same model: usual 60 with the cap 50 (19 / 17 / 14), set 2 46, check set
  43. A gap of more than 3 is looked into before anything is published, as section 9 said. Whatever comes out, the
  card reports the bf16 number next to the Mac's.
- The same pipeline ran on this kind of VM for runs 1-2's guard (`cache/guard/`), and on 11-08 vLLM gave ~2,000
  tokens/s on G4 for the cut models, so a set should take minutes, not hours.
- **Result (11-15 03:43; `cache/g4_bench/`).** All three sets were answered and graded; the session stopped with 3.75
  units left (export 23 min, vLLM 0.33.0 up 7 min later, then 21 min for the three sets; no padding expert was
  ever picked). **The cap did not run**: `think_cap.py` looked for answers starting with `<think>`, which the Mac's
  server echoes and vLLM does not, so it took all 180 as unthinking and capped none ("capped 0/60"; 10, 20 and 13
  answers are over 8,192 thinking tokens). Fixed with `--no-think-tag` (now in `colab_bench.sh`); the capped bf16
  numbers need a G4 again (export + load + the cap pass, about 5 units). So the comparison is without the cap:

| set (no cap, 16k, loose) | bf16, G4 | Mac 4-bit | only bf16 / only Mac | cut off at 16k (bf16 / Mac) |
|---|---:|---:|---:|---:|
| usual 60 | **48** (16 / 15 / 17) | 45 (17 / 15 / 13) | 9 / 6 | 8 / 12 |
| fresh set 2 | **49** (16 / 15 / 18) | 44 (14 / 13 / 17) | 8 / 3 | 10 / 11 |
| check set | **43** (13 / 16 / 14) | 43 (14 / 18 / 11) | 4 / 4 | 10 / 7 |

- 140 against 132 of 180; 21 questions right only in bf16, 13 only on the Mac (sign test p = 0.23). Set 2 differs by 5,
  more than the 3 section 9 allows; looked into: the export did not break the model (no padding expert picked, and
  it is the bf16 side that is higher), and the per-question split is within sampling noise. The direction agrees
  with the 54 T4 answers: the Mac's 4-bit numbers are, if anything, low.
- The goal's number, 46 of the usual 60, is reached in bf16 **without** the cap (48). Without the cap the Mac had 45.

## 10. Where is the cut model short? (design written 2026-11-14 01:50, before the run; the author: "find what is lacking, on the Mac")

The benchmark sets of this project are three subjects (science, math, code). The outside sets of section 9 cover more:
MMLU-Pro has 14 subjects (law, history, economics, psychology, health, philosophy, engineering, ...), IFEval checks
whether instructions on the form of the answer are followed (length, case, keywords, format). Overnight on the Mac:

- Sets: MMLU-Pro 280 (20 a subject, none this project has used) and IFEval all 541; thinking off, 2,048 tokens,
  the usual sampling. Thinking off keeps the night short and shows what the model knows without working it out.
- Models: the current cut + old LoRA, Qwen3.5-9B, the cut + run 2 LoRA, Qwen3.5-4B (Mac 4-bit, in that order).
- Grading: MMLU-Pro on the Mac (letter in the last box); IFEval with Google's checker on a Kaggle CPU kernel.
- Reading: a subject or an instruction kind is "short" if the cut model is 5 or more of 20 under 9B there (MMLU-Pro),
  or 10 points of percent under on an instruction kind with 30 or more cases (IFEval). Nothing is tuned on these sets;
  if one training run follows, its data comes from elsewhere, and these sets are only for the before and after.
- Addendum 04:20 (after seeing the cut model's IFEval answers, before 9B's): both sets are graded on the Kaggle CPU
  kernel (`heal/grade_ext_on_kaggle.sh`), IFEval now also by instruction kind. The cut model answered only 3 of 31
  "answer in language X" prompts in that language (Kannada asked, English given; Punjabi asked, Chinese with
  Vietnamese words given). IFEval's 31 prompts are few and one kind, so a probe of our own was added
  (`bench/lang_probe.py`): 20 languages with their own script, 2 everyday requests each (40), asked in English;
  passed when 80% of the letters are in that script. Same models, thinking off, 1,024 tokens. The queue was reordered
  (`cache/queue_gap2.sh`) so the probe runs right after 9B's IFEval. Reading: the language gap is real if the cut model
  is 8 or more of 40 under 9B on this probe and also short on IFEval's language kind by the rule above.

## 11. One training run for general knowledge (design written 2026-11-14 05:50, before the run; `heal/wiki_text.py`, `heal/colab_wiki.sh`)

- Why: section 10 found the cut model short on MMLU-Pro in 3 subjects by the rule written before it (economics
  7 against 9B's 16, engineering 4 against 9, health 7 against 12; 124 against 162 of 280 in all, thinking off).
  The misses read as wrong concepts ("regressive: the rate rises with income"), not as format. This is the same loss
  as Wikipedia perplexity 90 against 9.4 (6o) and the lost facts (6l). The heal data so far is code, math, science
  and UltraChat answers; no plain human-written text.
- What: run 2 again (same start, the 6M LoRA; same data `traces_mix2.jsonl`, same UltraChat share, rate and LoRA
  targets) with plain Wikipedia text added at 40% of the tokens at every point (`heal_train.py --text`). Loss on every
  token of a piece (pieces of up to 2,048 tokens from the start of an article). Half English and Japanese, half spread
  over 22 other large Wikipedias (a fixed list by size and script). Articles whose first 64 tokens match a stored
  Wikipedia window (C, A, B: where KL and perplexity are measured) are skipped. About 5.4M tokens, about an hour on G4.
  Comparing with run 2 shows what the text alone did.
- Measured during the run: KL on the chat windows (as always) and on the Wikipedia windows A+B at each milestone.
  Stop and do not use the result if the chat KL ends above 0.1464 (where run 2 started).
- After (Mac 4-bit): Wikipedia / chat / general perplexity (`mlx_ppl.py`); MMLU-Pro 280 and IFEval 541 (section 10's
  sets, thinking off, before and after only); the language probe; the usual 60 with the 8,192 cap; the 60 facts.
- Adopted as the new current model if all hold: (1) usual 60 with the cap at least 47 (the current model has 50;
  3 is the noise allowed so far); (2) MMLU-Pro up by 10 or more of 280 over the current model (124); (3) Wikipedia
  perplexity down by a third or more (from 89.6); (4) IFEval not down by more than 10 of 541. Otherwise the result is
  reported and the current model stays. Nothing is tuned on these sets; there is one run.
- **Addendum 06:45 (part L), after 9B's IFEval and the first half of the language probe, before any training
  result.** IFEval's "answer in language X": 9B 31 of 31, the cut model 3. Language probe (the first 20 of 40; the
  other 20 were left out by `eval_chat.py`'s default of 20 a task, to be asked later): 9B 20 of 20, the cut model 1.
  Even its Japanese mixes in Chinese ("天空は蓝色的是因为…"); Russian and Hindi come back in English, Arabic and Kannada
  in Chinese. Plain text alone may not bring back answering in a language, so the run that had started (step 5 of
  about 160) was stopped and replaced by one that also has the original's own answers in the 23 languages
  (`heal/lang_data.py`, `heal/colab_wiki_l.sh`): 40 prompts per language written by people in that language (the
  Aya dataset) and 15 English UltraChat prompts with a request to answer in it (worded unlike the probe and
  IFEval), half with thinking on (4,000 tokens) and half off (1,500); finished answers only. Everything else is as
  above. Reading of the language probe: languages in this list (ja zh ko ru ar hi bn ta th he el) are reported apart
  from those not in it (te kn gu pa ml ka hy am si); the probe's form is near the training form, IFEval's language
  kind is the less near check. Added to the adoption rule: (5) IFEval's language kind up by 10 or more of 31.
- Not done here: LoRA on the routed experts (the Mac adapter path does not take it), a teacher pass over the text
  (the original's distribution instead of the plain next token; it needs the original and the student together).

### Section 11 result, part 1 (2026-11-14 07:32–08:53 Colab G4; `cache/honesty/wikil/`, adapter `cache/heal_adapters/wikil`)

- Data as fed: run 2's rows + 1,451k tokens of the original's answers in the 23 languages + 1,600 Wikipedia pieces
  (1.69M tokens: en 377k, ja 406k, about 40k each for the other 22). The text was shorter than planned, so it ran out
  at about 4.2M of the 7.0M tokens: 24% of all tokens, not 40% (the loss falls from about 2.1 to 1.0 there). The run
  ended with the data (step 193), not with the time limit.
- During the run: chat KL A+B 0.1464 at the start, 0.1426 at 2M, 0.1431 at 4M, 0.1389 at the end (below the stop line,
  so the result is usable). Wikipedia KL A+B 2.264, 1.794, 1.696 (not measured at the end).
- Perplexity (Mac 4-bit, `cache/ppl_wikil.json`): Wikipedia 43.8 (from 89.6, more than halved: rule 3 holds), chat
  1.856, general 10.0.
- Language probe (all 40, thinking off, 1,024 tokens): 14 of 40 in the language asked; the old model had 1 of the
  first 20, 9B 20 of 20. All 14 are languages in the training list (14 of 22 of those prompts). None of the 18 prompts in
  languages not in the list passes. The answers are still poor: 22 of 40 run to the limit, many repeat a word ("寝事を寝事を…"),
  and the Japanese that passes has wrong words ("光は非常に快的スピード"). The probe checks the script only, so 14 is
  more than the real ability.
- The usual 60, MMLU-Pro, IFEval and the facts: running on the Mac (`cache/queue_cand.log`).
- **MMLU-Pro (14:10): 113 of 280, against 124 for the current model** (graded on Kaggle,
  `cache/gap/grades_cut_wikil_mmlu/`). Rule 2 asked for +10; it is -11, so the run is **not adopted** whatever the other
  measures show; the current model (6M LoRA) stays. By subject (new / current / 9B, of 20 each): economics 9/7/16,
  engineering 6/4/9, health 3/7/12, history 2/5/9, computer science 10/13/12, the rest within 2. The three subjects
  section 10 found short moved +2, +2, -4. Halving the Wikipedia perplexity did not bring back answering the questions:
  the text taught the form of encyclopedic text more than the facts the questions ask for. The other measures still run,
  for the next design (whether the language answers moved IFEval's language kind; whether the usual 60 held).
- The facts probe (section 6l's 110 prompts, thinking off, 1,024 tokens, as every run of it; `heal/halluc_probe.py grade`):

  | answers | made-up names described / denied (of 40) | real described (of 10) | facts right / wrong / abstains (of 60) | wrong of answered |
  |---|---|---|---|---|
  | current (6M LoRA) | 40 / 0 | 10 | 11 / 46 / 3 | 0.81 |
  | run 2 (honest2) | 21 / 19 | 8 | 12 / 38 / 10 | 0.76 |
  | this run (wikil) | 28 / 12 | 9 | 12 / 31 / 17 | 0.72 |
  | 9B | 17 / 23 | 10 | 60 / 0 / 0 | 0.00 |

  The run's honesty rows held part of run 2's gain (12 made-up names denied, 17 abstains), but facts right stayed at 12
  of 60: like MMLU-Pro, the Wikipedia text did not bring the facts back.
- The 40-prompt language probe on the same day's runs: current model 2 of 40 (the 1 of 20 above plus one Greek
  answer), wikil 14, 9B 40.
- The 8,192 cap of the usual 60 stopped at 38 of 60 twice: with other apps open the Mac had 4-9% free, and the memory
  guard killed the server (8 times in stage CAND, 30 times at 15:12-15:20). It is queued again after stage GAP and starts
  only with 30%+ free (`cache/after_gap.sh`); the usual 60 (16k), IFEval and the rest of stage GAP are graded on Kaggle
  in the same script.
- **Correction (17:10): the usual 60 was not finished either.** `heal/mac_eval.sh` ran out of its 30 attempts at 12:28
  with 24 questions still unanswered (all 20 MMLU-Pro and 4 math: the server was killed by the guard each time those
  long answers grew), wrote `== done` anyway, and the 24 error rows were graded as wrong. Only 36 were answered: math
  16 of 16, code 19 of 20 (Kaggle). The script now says `== gave up` and fails (no other answer file had error rows).
  The cap at 16:54 started with 45% free and was killed 10 times at 5-7% free, so this model's long answers do not fit
  next to the apps that were open; rule 1 is unmeasured.
- IFEval (Kaggle, `cache/gap/grades_batch2/`): wikil 369 of 541, current model 345 (rule 4 holds, +24); the language
  kind 15 of 31 against 3 (rule 5 holds, +12). Run 2 (honest2) 355 and 4 of 31; 9B 436 and 31; 4B 407 and 31.
- MMLU-Pro of the same batch: run 2 118, 4B 176 (9B 162, current 124; all thinking off, 2,048 tokens). 4B above 9B
  is unexpected and not yet looked into.
- So rules 3, 4 and 5 hold, rule 2 fails and rule 1 is unmeasured: **not adopted**, as decided at 14:10.
- Paired check of MMLU-Pro (18:40, the same 280 questions graded on the Mac with `bench/ext_bench.py`'s rule): 30
  right before and wrong after, 19 the other way (exact two-sided sign test p = 0.15). So the -11 is within the noise:
  the run did not raise MMLU-Pro, rather than lowered it. Rule 2 (+10) fails either way.
- Rule 1 (23:40): the usual 60 with the 8,192 cap, after the reboot, with the 6 server-error rows redone one at a time
  (`cache/cap_wikil_fix.sh`, 0 error rows left; `heal/think_cap.py` no longer keeps error rows of the source answers):
  **53 (15 / 19 / 19)** against the current model's 50 (rule 1 holds). With rule 2 failing, wikil is still **not
  adopted**; the +3 is inside the noise of 60 questions and is not a reason to change the 14:10 decision.
- **Why MMLU-Pro fell (23:45, from the stored answers, `cut_old` against `cut_wikil`, 280 questions, 2,048 tokens):**
  30 lost, 19 gained. 13 of the 30 lost ran out of the 2,048 tokens before the boxed letter (8 of the 19 gained were
  the other way): answers cut off rose from 61 to 75, the mean length from 1,013 to 1,092 tokens, loops among the cut
  answers from 7 to 12. By subject the losses sit in recall-heavy ones (health 7 → 3, history 5 → 2, law 5 → 3,
  computer science 13 → 10) and the gains in worked ones (economics 7 → 9, engineering 4 → 6); 20 questions a
  subject, so no single subject is more than noise.
- **Re-measure at 4,096 tokens (rule written 23:53, before the run; the author asked for it):** the same 280 questions,
  thinking off, Mac 4-bit, current model (`heal_adapters/6e6`) and wikil, after the guard runs
  (`cache/mmlu4k.sh`). It does not change the decision above or README 14's tie-break (both were set at 2,048); it
  decides the next Wikipedia design:
  - if wikil is at least the current model minus 3 at 4,096 (the length explains the fall), the next run keeps the
    Wikipedia data and adds chat traces to the mix so the answer length stays where it was;
  - if wikil is still more than 3 below, the fall is not only length: the next run also lowers the learning rate or
    the Wikipedia share, and the recall subjects are read first.
  - Reported either way: totals, answers still cut off, the paired counts with the sign test, and the subjects.
- **Result (11-15 09:18, graded on the Mac with `bench/ext_bench.py`'s rule; `cache/gap/cut_{old,wikil}/mmlu4k/`):**
  - Current model **132** (124 at 2,048), wikil **121** (113). Still 11 below, so **the second branch applies**: the fall
    is not only length. A next Wikipedia run lowers the learning rate or the Wikipedia share.
  - Cut off at 4,096: current 44, wikil 63. Mean tokens: 1,406 and 1,630. wikil's answers are still longer.
  - Paired: 26 right only for the current model, 15 only for wikil (sign test p = 0.12).
  - Subjects that fell most (current / wikil, of 20): health 5/2, history 3/1, chemistry 11/8, biology 19/17,
    computer science 14/12, physics 14/12. Recall subjects fall again, as at 2,048.

## 12. Do the made-up answers change from sample to sample? (design written 2026-11-14 18:15, before the run; `heal/consist_probe.py`)

The user wants to publish but not with answers that describe functions that do not exist (the facts probe: the current
model describes all 40 made-up names). The user asked whether a reward for giving the same answer over many samples
could fix it. Rewarding sameness itself also rewards the same wrong answer every time, and "I don't know" every time
scores best; so first, with no training, measure whether disagreement between samples marks what the model does not
know (as in the semantic entropy work). If made-up answers vary and right ones agree, "answer only when the samples
agree" works at inference time and can later be taught by SFT on questions outside the probes; if wrong answers are as
stable as right ones, the knowledge is gone and abstaining cannot fix it.

- Model: the current model (g128b28160 + 6M LoRA), Mac 4-bit, thinking off, 1,024 tokens, the default sampling with no
  seed, 8 samples per prompt (8 runs of `heal/mac_eval.sh`, `cache/consist/s1..s8/`).
- Sets: the facts probe (40 made-up names, 10 real, 60 facts; `fix_task/halluc_set.jsonl`) chooses the thresholds;
  the second fact set (60 facts never used to choose anything; `fix_task/halluc_set2.jsonl`) confirms the facts part.
  There is no second set of made-up names, so that part is chosen and read on the same prompts and is reported as such.
- Agreement: facts, the share of the 8 short answers equal to the most common one (lower case, punctuation and
  articles dropped, first 6 words); names, the mean pairwise overlap (Jaccard) of the content words of the 8 answers
  (words of 4+ letters, without the question's own words). The kept answer is sample 1.
- Thresholds (chosen on set 1 only): facts, the agreement that maximises right kept minus wrong kept; names, the overlap
  that maximises made-up names refused plus real names kept.
- Worth teaching to the model (an SFT design, written before that run) if both hold: (1) on set 2 the wrong share of the
  answered facts falls by a third or more from sample 1 with no filter (the earlier single run `cache/halluc2/
  g128b_heal6e6` is shown next to it) while at least 80% of sample 1's right answers are kept; (2) on set 1, at least 20 of 40 made-up names refused with at least 8 of 10 real ones kept.
  If (1) fails because wrong answers agree as much as right ones, the facts are lost, not uncertain.
- Then, if there is time: the FaithDial check (does the model keep to a given text; section 13, to be written first).

### Section 12 result (2026-11-14 19:51-20:07, Mac; `cache/consist/`, `cache/after_reboot2.log`)

- All 8 samples answered all 170 prompts (no server errors; the samples differ, e.g. `itertools.pairmerge` is described
  three different ways in samples 1-3). Mean agreement by sample 1's verdict:

  | | facts set 1 | facts set 2 |
  |---|---|---|
  | right | 1.00 (n 10) | 0.75 (n 16) |
  | wrong | 0.38 (n 43) | 0.35 (n 37) |
  | abstains | 0.55 (n 7) | 0.61 (n 7) |

  Most wrong facts change from sample to sample; only 1 of 43 (set 1) and 3 of 37 (set 2) are the same in all 8. So
  the wrong facts are mostly not known, rather than known wrong.
- Rule (1) fails. Set 1 chose agreement 1.00 (all 10 right answers there were unanimous). On set 2 that keeps 7 of 16
  right (44%, under 80%); wrong share 0.70 to 0.30 (that half holds). The threshold fit set 1's few right answers too
  tightly. Shown only to read the result, not to choose anything (chosen on set 2 itself): 0.375 keeps 15 of 16 right
  and cuts wrong from 37 to 14 (share 0.48); 0.625 keeps 11 and leaves 6 wrong (0.35).
- Rule (2) fails. Word overlap does not separate made-up names (mean 0.15, 0.07-0.37) from real ones (0.23,
  0.10-0.36): real descriptions are worded differently each time too. The threshold refuses 38 of 40 made-up names but
  keeps only 4 of 10 real ones. Word overlap is too crude here; meaning would have to be compared.
- Reading: for facts the signal is strong (right answers agree, wrong ones scatter), so teaching "answer only when sure"
  is worth a designed run, with its threshold chosen on a larger set of new questions. For function names the truth is
  known without the model (Python can check whether a name exists), so training rows there need no agreement at all.
  Not adopted as an inference-time filter by the rules above.

## 13. Run 3 against made-up answers: facts split by agreement (design written 2026-11-14 20:40, before any run)

The user, after section 12: build the training data in that direction, on Colab. Section 12 found that the cut
model's wrong facts mostly scatter from sample to sample (not known) while a few repeat (known wrong). Runs 1 and 2
taught "I don't know" for every fact the cut model missed, including the ones it repeats wrong.

- **Labels** (`heal/honesty_data.py label3`, from the stored 5 samples of the current model on the 1,326 facts the
  original wrote in run 1, `cache/honesty/student/`; no new model calls): known, 4-5 of 5 right (123); **corrected**,
  0-1 right with one wrong answer 3+ times of 5 (193; trained with the original's right answer); unknown, 0-1 right
  and no wrong answer more than twice (676; trained with "I don't know" as before); 2-3 right dropped (66). The
  threshold (3 of 5 the same) follows section 12's reading, not a value tuned on any probe.
- **Mix** (`build --labels facts_labeled3.jsonl --cap 400 --passes 2 --out-name traces_mix3.jsonl`): the run 2
  things (all stored answers of the original) and guard data unchanged; facts up to as many "I don't know" as right
  answers (known plus corrected), 400 at most. After the existing cap (honesty answers no more than the code answers):
  113 known, 172 corrected, 285 "I don't know" (run 2: 123 each of known and "I don't know"); tokens a third each for
  honesty, code, math with science; 1,808 examples over two passes. No new question forms (the honesty share is
  already full: 1,479 rows were cut by the cap).
- **Run**: Colab G4, one session (`heal/colab_run3.sh`): the student cut on the VM, training as run 2 (LoRA from the
  6M LoRA, same targets and rate). Changed at 20:45 before the start: only 29.3 compute units are left, so the export
  and the Colab guard are dropped (about 1 hour instead of 1.5-2) and the guard runs on the Mac, in README 8's own form.
- **Adopted as the current model if all hold** (all on the LoRA from the end of the run):
  1. Guard (check set, Mac 4-bit, 8,192 cap, as README 8 first wrote it; before: code 11, total 43, science 13): code
     not down by 2+, total not down by 3+, science not down by 2+.
  2. Facts, both probes together (120): right not down by more than 2, and wrong of answered down by 0.10 or more
     (before 0.76; run 2 reached 0.74).
  3. MMLU-Pro (Mac, 280) not lower by a paired sign test at p < 0.05 against the current model's answers, and IFEval
     not down by more than 10 of 541.
  The transfer probe, the made-up names of 6j and section 12's agreement are reported, not ruled on: this run adds
  nothing aimed at them.

### 13a. wikil-a and wikil-b (design written 2026-11-15 00:05, before any run; the author: "if units are left")

The user would like the Wikipedia run (wikil, section 11) to get the same treatment: **wikil-a** = wikil plus run 3's
training against made-up answers; **wikil-b** = wikil-a plus a fix for its longer answers.

- **wikil-a**: run 3's data (`traces_mix3.jsonl`) and settings, started from wikil's LoRA (`cache/honesty/wikil/
  lora_final.pt`) instead of the 6M LoRA, at most 0.8 hours (`heal/colab_wikila.sh`). It runs in run 3's session after
  run 3 is fetched, and only if at least 11 units are left then (G4 about 10 an hour); otherwise it is skipped and the
  session stops (`cache/colab_run3_go2.sh`). The wikil LoRA is uploaded while run 3 trains.
- **Judged by run 3's three rules** against the current model (guard, facts, MMLU-Pro paired / IFEval), and it joins
  README 14's candidates. Reported, not ruled on: MMLU-Pro answers cut off at 2,048 and the mean length against wikil
  (75, 1,092) and the current model (61, 1,013). Run 3's mix holds the original's own code, math and science answers,
  so it may pull the length back by itself; that is what the report shows.
- **wikil-b is not designed yet**: it waits for section 11's 4,096-token re-measure (is the fall the length?) and
  wikil-a's length. If wikil-a's answers are back to the current model's length, wikil-b is not needed. If it is
  needed, it is a separate session (about 10 units with the student cut again), which the units left after run 3 and
  wikil-a are unlikely to cover; 200 more units come on 11-23.

### 13b. Results of run 3 and wikil-a (2026-11-15; Mac 4-bit, LoRA from the end of each run)

- **Facts, both probes together (120; thinking off, 1,024 tokens; `heal/halluc_probe.py grade`, `cache/halluc*/`)**:

  | answers | right | wrong | abstains | wrong of answered | made-up names described / denied (of 40) |
  |---|---:|---:|---:|---:|---|
  | current (6M LoRA) | 26 | 83 | 11 | 0.76 | 40 / 0 |
  | run 2 (honest2) | 26 | 74 | 20 | 0.74 | 21 / 19 |
  | run 3 | 25 | 56 | 39 | **0.69** | 27 / 13 |
  | wikil-a | 19 | 40 | 61 | **0.68** | 25 / 15 |
  | wikil | 26 | 62 | 32 | 0.70 | 28 / 12 |

  Per set (right / wrong / abstains): run 3 11/31/18 and 14/25/21; wikil-a 9/20/31 and 10/20/30; wikil 12/31/17 and
  14/31/15 (its second set run 11-15 07:20, `cache/queue_new2.sh`).
- **Rule 2 fails for both, so neither is adopted as the current model.**
  - run 3: right 25 (-1, within the 2 allowed), but the wrong share fell by 0.07, not the 0.10 asked.
  - wikil-a: the share fell by 0.08 and right fell by 7.
  - Both moved the share further than run 2 (0.74), mostly by abstaining more, not by knowing more.
- Transfer probe (45, reported only; `honesty_data.py grade3`): non-existent things called out, of 30: current 5,
  run 2 12, run 3 12, wikil-a 11, 9B 16. Real things answered, of 15: 15 each (current 14).
- Guard (check set, 8,192 cap, code graded on Kaggle, `cache/gap/grades_guard_cand/`), for README 14:
  - run 2 (honest2) **44** (science 14, math 17, code 13).
  - wikil **43** (12, 17, 14).
  - Both hold the guard against the current model (43; science 13, code 11).
  - run 3 and wikil-a are queued (`cache/queue_new2.sh`), after the MMLU-Pro 4,096 re-measure.
- README 14 so far: wikil-a (0.68) and run 3 (0.69) have the lowest wrong share. They are within 0.03, so MMLU-Pro
  decides if both hold the guard (wikil, 0.70, is third). Both guards are still to come.
- MMLU-Pro at 4,096 tokens (README 11's re-measure), current model, graded on the Mac (`bench/ext_bench.py`'s rule):
  **132** of 280 (124 at 2,048), cut off 44 (61), mean 1,406 tokens (1,013). wikil's run follows.
- run 3's guard (11-15 12:19, `cache/gap/grades_guard_run3/`): **44** (science 13, math 17, code 14); it holds.
- run 3's MMLU-Pro (2,048, thinking off, Mac grading, 13:57): **111** of 280 against 124. Paired: 29 right only
  for the current model, 16 only for run 3 (p = 0.07), so rule 3's "not lower at p < 0.05" holds, narrowly. Cut off
  at 2,048: 80 (current 61), mean 1,136 tokens (1,013): longer answers, as with wikil.
- Japanese samples (13:57-14:03, the author wanted to read the Japanese; `fix_task/ja_sample.jsonl`, 2 prompts written
  that day, one answer each, `release/ja_samples_2026-11-15.md`). **Both models fail in Japanese:**
  - run 3, thinking on: both answers thought in English to the 4,096 limit and gave no answer. Thinking off: both
    answers in Chinese (the arithmetic, 7, right).
  - wikil-a, thinking on: the arithmetic answer right (7) in odd Japanese; the other turned into one character
    repeated. Thinking off: both broke into repeated characters, and the arithmetic answer said 3.
  - The card's warning stands and must be stronger: Japanese is not usable without a translation step.

## 14. Which model goes out as the preview (rule written 2026-11-14 21:25, before run 3's results)

The user's plan: finish the evaluations, train run 3 on G4, publish the best model as a preview unless it is too bad
to show. Candidates: the current model (6M LoRA), run 2 (honest2), the Wikipedia run (wikil), run 3 (and wikil-a, added
00:05 on 11-15 before any of its results, section 13a).

- A candidate must hold README 8's guard against the current model (check set, Mac 4-bit, 8,192 cap: code not down
  by 2+, total not down by 3+, science not down by 2+); the current model holds it by definition.
- Among those, the one with the lowest wrong share of answered facts on both probes together (120 facts) goes out:
  made-up answers are what holds the release back. If the best two are within 0.03, the higher MMLU-Pro (Mac, 280)
  decides.
- Models are chosen on the Mac; the exact 2x T4 benchmark (section 9) is run for the chosen one only (one run took
  7+ hours of the weekly Kaggle GPU time).
- The model card leads with the weak points, in numbers: the facts probe, made-up names, MMLU-Pro and IFEval next
  to 4B and 9B. Publishing is the author's action.
- **Decision (11-15 18:28), by the rule above:**
  - Guard: run 3 44 (13 / 17 / 14), wikil-a **44** (13 / 18 / 13); both hold. honest2 44 and wikil 43 hold too.
  - Wrong share of answered facts (120): wikil-a **0.68**, run 3 0.69, wikil 0.70, honest2 0.74, current 0.76.
  - The best two are within 0.03, so MMLU-Pro (Mac, 280, thinking off) decides: wikil-a **123**, run 3 111.
  - **wikil-a goes out as the preview.** Its MMLU-Pro against the current model: 123 vs 124, paired 20 / 21.
- **Uploaded (11-15 20:29, private):** the Kaggle run folded wikil-a's LoRA (250 linears, mean relative change
  0.038) and checked it: KL to the original on the 60 held-out windows 0.1424 against the run's own 0.1426. The
  uneven checkpoint (18.7 GB) and the padded one (37.1 GB, 582 dummy experts, 0 dummy hits) are on Hugging Face as
  private repos. (First try failed at the upload: Kaggle's newest huggingface_hub has no `upload_large_folder`;
  pinned to 1.33.)
- Language probe (the first 20 of the 40 prompts, thinking off): current 1, run 3 1, wikil-a **9** (wikil 14 of 40).
- **Release steps (11-15, the author: "once the model is chosen, start the upload without benchmarking"):**
  - Mac queue reordered (`cache/queue_new3.sh`): both guards and both MMLU-Pro first, then the IFEval runs.
  - The Kaggle notebook `heal/kaggle/release` (2x T4) does the rest. It folds the chosen LoRA into the uneven
    checkpoint shard by shard (`heal/fold_lora_uneven.py`). It stops unless the folded model's KL to the original on
    the held-out windows is within 0.01 of the run's own end-of-run number. Then it uploads, as **private** repos:
    - the uneven checkpoint, with `uneven.py`;
    - `pad_uneven.py`'s plain checkpoint, built from the folded weights (stops if a dummy expert would be picked).
  - The HF token is a Kaggle secret, given only to runs the author starts from the editor ("Save Version"). Making the
    repos public is the author's action.
  - Not in this step:
    - The MLX 4-bit build needs about 30 GB free on the Mac (13 GB now).
    - The exact bf16 benchmark of the chosen model waits for Colab units (11-23). Until then the card gives Mac 4-bit
      numbers, and the G4 bf16 numbers only for the model they were run on.
    - No transformers remote code exists for the uneven shape yet. The card must not claim
      `trust_remote_code=True` until it does.

## 15. Outside benchmarks on the G4 (design written 2026-11-15 13:20, before any run; the author added 100 units)

The user: measure the chosen model next to NeoHorse-1-9B, Qwen3.5-9B and Qwen3.5-35B-A3B on AIME 2026, MMLU-Pro and
LiveCodeBench v6. `bench/ext_bench.py` (11-13) has the sets; `bench/colab_ext.sh` runs them on one Colab G4.

- **Sets** (`cache/ext/`, made 11-14):
  - AIME 2026: 30 problems × 3 samples (score = mean right of the three).
  - MMLU-Pro: 280, 20 per subject. This set was used thinking off on the Mac to judge runs (README 10, 11, 13, 14),
    so it is not fresh for this project's own runs. It is fresh for the three other models.
  - LiveCodeBench v6: the 175 problems added in release 6, all public and private tests must pass (run on the VM).
- **Settings, the same for all four**:
  - vLLM, bf16, one G4.
  - Thinking on, temperature 0.6, top-p 0.95, top-k 20, no seed, up to 32,768 new tokens, no thinking cap.
- **Models**:
  - TokenRhythm/NeoHorse-1-9B, Qwen/Qwen3.5-9B, Qwen/Qwen3.5-35B-A3B (the original), from Hugging Face.
  - Ours: the padded bf16 checkpoint (`moe/pad_uneven.py`, dummy check as README 9).
  - Run 3 and wikil-a are both measured here, since the G4 would otherwise wait for the Mac's choice. **These numbers
    do not change README 14's choice**, which stays on the Mac's rules. The card reports the chosen one; the other
    is listed in this README.
- **Order**: the three references first; then the student cut, the two exports and both sets. Then, for the chosen
  model only, the usual 60, set 2 and the check set with the 8,192 cap (README 9b, with the fixed cap script).
- **Reported as they come**: every model, every set, nothing re-run or dropped after a score is seen. A set not
  finished before the units run out is reported as unfinished. If the vLLM version or a setting changes between
  models, that is said next to the numbers.
- **Budget**: about 7 hours of G4 (about 70 units). The session stops when the balance falls under 15 units.
- AIME 2026 came out in February 2026. Whether the reference models saw it in training is not known.
- **Added 13:40, before any score (the author asked about IFEval):** IFEval (all 541 prompts), thinking on, the same
  settings, for all five models, after everything above (`bench/colab_ifeval.sh`). Graded on the VM with Google's
  checker (`lm-eval[ifeval]`), strict, plus the language kind. It runs only if at least 30 units are left at that
  point; otherwise IFEval stays the Mac's thinking-off numbers (README 10), said as such.
- **Changed 14:35 (the author; the units do not cover the plan: Qwen3.5-9B took about 1 h 45 min, not 1 h, at about
  9.6 units an hour, 86 units left).** No score of run 3 or wikil-a had been seen; Qwen3.5-9B's were partly in.
  - G4 order: Qwen3.5-9B (as running) → the student and both exports → **MMLU-Pro only** for run 3 and wikil-a →
    NeoHorse-1-9B → then the chosen model's AIME and LiveCodeBench and Qwen3.5-35B-A3B, the chosen model first if
    README 14 has decided by then.
  - The other of run 3 / wikil-a gets no AIME or LiveCodeBench.
  - IFEval (thinking on, same settings) moves to Kaggle 2x T4, fp16, for Qwen3.5-9B and NeoHorse-1-9B
    (`heal/kaggle/ifeval_ref`). The 35B original and our padded model do not fit on two T4s. Their IFEval is asked
    last in their own G4 runs (the model is already loaded); if the units run out first, it is reported as not
    measured.
  - The cut-off moves from 15 to 8 units so the chosen model can finish.
- **Results so far (G4, bf16, thinking on, 32,768 tokens; `cache/g4_ext/`):**

  | model | AIME 2026 (of 30) | MMLU-Pro (of 280) | LiveCodeBench v6 (of 175) | MMLU-Pro cut off at 32k / mean tokens |
  |---|---:|---:|---:|---|
  | Qwen3.5-9B | 22.3 | 226 | 87 | 3 / 5,626 |
  | run 3 | – | 135 | – | 56 / 11,623 |
  | wikil-a | – | 144 | – | 41 / 10,507 |

  - With thinking on, our two MMLU-Pro numbers are above their thinking-off Mac numbers (run 3: 111 at 2,048).
  - A fifth to a sixth of their answers loop to the 32k limit without answering (looping flagged in 29-30).
  - These do not enter README 14's choice.
- **Later results (11-15, G4 unless said):**

  | model | AIME 2026 | MMLU-Pro | LiveCodeBench v6 | IFEval (strict) |
  |---|---:|---:|---:|---:|
  | wikil-a (the release) | 20.7 | 144 | 68 | 362 |
  | NeoHorse-1-9B | 20.3 | 221 | 96 | (Kaggle, running) |
  | Qwen3.5-9B | 22.3 | 226 | 87 | (Kaggle, running) |
  | Qwen3.5-35B-A3B | unfinished | unfinished | unfinished | unfinished |

  - wikil-a cut off at 32k: AIME 27/90, LCB 95/175, IFEval 135/541.
  - The 35B original was stopped at 22:15 when the units ran out (the author chose not to add more). Answered by then:
    AIME 82/90, MMLU-Pro 116/280, LCB 77/175, IFEval 123/541 (`cache/g4_ext/orig/*.partial`).
  - On the same answered subsets (the questions it finished first, so likely the easier ones; not a full score):
    AIME (82 answers over 28 problems) original 26.0, Qwen3.5-9B 22.0, wikil-a 21.3, NeoHorse 20.3. MMLU-Pro (116)
    original 108, Qwen3.5-9B 105, NeoHorse 104, wikil-a 78.
  - GGUF Q4_K_M (10.6 GB) in the main repo, checked with llama.cpp on a CPU on three prompts (all sensible).

## 6k. Does the cut model work as an agent? (design written 2026-11-13, before any run; `heal/agent_probe.py`)

- Why (the author's question): a local agent app drives a model through tool calls, many turns in a row. Answering a
  benchmark question is not that. The app on this Mac loads models through LM Studio, which cannot read the uneven
  shape (the number of experts differs by layer), so the test goes through our own server with the same tool format
  (the model's chat template, mlx-lm's parser for it). What 6h found makes this a real risk: an agent must copy tool
  names, argument names, file names and ids exactly.
- Tasks: 24, written for this test. Eleven tools over a made-up workspace held in memory (files, orders, messages;
  nothing on the Mac is touched): look-ups, file moves and edits, order updates, a message with a given text, a
  three-step to-do list, a condition, and two requests that cannot be done (a file and an order id that do not
  exist; the id differs from a real one by one character). File names and ids are long on purpose.
- An episode: up to 12 turns, the usual sampling, thinking on, 4,096 tokens a turn. Solved = the workspace ends in
  the right state (and the final answer holds what was asked); how the model got there is not graded.
  `selftest` checks that every task is solvable and none passes untouched.
- Run: Mac 4-bit, 2 samples a task (48 episodes): 9B, `g128b28160` with each LoRA, 4B.
- Reported: solved of 48; how unsolved episodes end (wrong end state, turn limit, out of tokens, a tool call the
  parser cannot read, empty); episodes with a call to a tool or argument that does not exist, and with a "name
  slip" (a path, id or status within two edits of a real one, not written that way in the task).
- Reading: against 9B, differences under 6 episodes are not read. "Works as an agent like 9B" = solved within 6 of
  9B and no more than 6 episodes above 9B with an unreadable call, a wrong tool or argument name, or a name slip.

### Result (2026-11-13; Mac 4-bit, 48 episodes each, about 5 minutes a model; `cache/agent/`)

| model | solved (as designed) | solved if an answer left inside the thinking counts | episodes with a wrong e-mail or id in a call | unreadable call / unknown tool / wrong argument names |
|---|---|---|---|---|
| Qwen3.5-4B | 48 | 48 | 0 | 0 / 0 / 0 |
| Qwen3.5-9B | 44 | 47 | 0 | 0 / 0 / 0 |
| `g128b28160` + trained LoRA | 40 | 44 | 5 | 0 / 0 / 0 |
| `g128b28160` + old LoRA | 37 | 42 | 4 | 0 / 1 / 4 |

- **The tool-call format holds.** No model wrote a call the parser could not read; the cut models call the right
  tools with the right argument names in almost every episode and chain three to five calls without help.
- By the rule fixed above: the trained LoRA is within 6 of 9B (40 against 44) with no excess of format errors, so it
  "works as an agent like 9B" on these tasks; the old LoRA misses by one episode (37) and has 5 episodes with a
  wrong tool or argument name. 4B solves everything.
- The second and third columns were added after seeing the results, and say more than the rule does. Every one of
  9B's four misses but one is an episode whose work was right and whose final answer stayed inside the thinking
  (the reply shown to a user is empty); counting those, 9B has one miss (an extra file it created). The cut models'
  remaining misses are real errors: a copy done as a move and reported as "the original remains unchanged"; a
  config file overwritten with made-up content instead of the one-line change; an error count of 5 for 4; both
  September and October reports moved and described as "both October"; and the customer's address copied wrongly
  from the task ("jasen.weber", "jason.weber", "piriya.nair" for jonas.weber and priya.nair) in 4–5 episodes.
- The loop helps: after a wrong address the tool answers "no such customer", and in about half of those episodes the
  model then finds the right one and finishes. Where it does not, it reports that the customer does not exist.
- So: as an agent the cut model is usable on short tasks and its failures are the kind found everywhere else in
  this section — exact copying of names — plus false reports of what it did. That is worse for an agent than a lower
  score, because the report cannot be trusted. Small sample (24 tasks, 2 samples), short tasks only.

## 6i. Loops in the 96×256 answers (2026-11-11; stored answers, no new run; `heal/loop_report.py`)

A loop is the first 4,000-character window of the thinking that compresses below 15% (the grader's rule). 60 answers
each, thinking on, 16,000-token limit, no cap.

| model | closed, no loop | loop, never closed | loop, then closed | no loop, hit the limit | loop starts at (median tokens) |
|---|---:|---:|---:|---:|---:|
| 96×256 untrained (usual 60) | 2 | 57 | 0 | 1 | 420 |
| + LoRA 1M tokens | 22 | 33 | 0 | 5 | 3,100 |
| + LoRA 4M | 26 | 23 | 0 | 11 | 5,300 |
| + LoRA 6M, loop-exit data (usual 60) | 38 | 9 | 1 | 12 | 5,200 |
| + LoRA 6M (check set) | 36 | 12 | 1 | 11 | 2,900 |
| 4B (check set) | 36 | 13 | 0 | 11 | 5,500 |
| 9B (check set) | 41 | 2 | 0 | 17 | 12,700 |

- **How often**: untrained, 57 of 60 loop, half of them within the first 420 tokens. Healing brings it to 10–13 of
  60, the same count as 4B — but a further 11–12 answers run to the limit without a clean loop (4B 11, 9B 17).
- **Leaving a loop**: 2 of 23 looping answers of the healed model closed their thinking, none usefully (one wrote an
  empty answer after "tantantant…", one answered D where F was right). 9B and 4B: 0 of 15.
- **The loop-exit sentence was learned, the exit was not.** In 13 of those 23 answers the model writes lines like
  "I'm checking the same thing again. Let's stop and get the next check." — 8 to 30 times per answer — and goes on.
  Before the loop-exit data (1M, 4M) no looping answer says anything of the kind.
- **When** (the 23 loops of the healed model, read one by one): 10 are a calculation that does not come out and is
  retried the same way (8 math, 2 science); 6 re-check a settled answer ("So, C. Wait, …"; 4 science, 2 math);
  3 rewrite the same plan or code; 2 re-read the labels of a figure given as drawing code; 2 collapse into one
  repeated fragment ("-hand-hand-hand", "tantantant").
  Lost words show inside them: "Fluency" for the dispersion force, "Horizon's Rule", "Contriction", "acceled".
  Untrained, the trigger is cruder: the model cannot repeat words of the question ("largest_sm smallest integers",
  "divother"), calls them typos or "nonsense words", and circles on that.
- **What a loop costs** (healed, check set, no cap): closed without a loop 24 of 36 right; looping 2 of 13; at the limit
  without a loop 2 of 11. 4B: 34 of 36, 5 of 13, 0 of 11. So the healed model loops no more than 4B; it loses mainly
  on the answers it does finish (24 of 36 against 34 of 36), above all code (10 of 18 against 15 of 15).

## 6f. WeightFix middle tasks and ranking (CPU stage; details in `../DESIGN.md`, sections v1c and v1d)

- **Decoys** (one or three harmful outliers plus six harmless ones of the same size) and **sign flips** (one or three
  ordinary-sized weights negated) were built with the same nets and splits as v1: 458 problems, 194 on held-out games.
  Only 8 held-out problems exist for a single flip (one flipped weight rarely costs 0.25), so flips are read on flip3.
- Decoys are a middle task only for the first try: reading magnitude gets `closed` 0.12–0.19 at budget 1 (oracle
  0.93–0.98) but 0.78–0.84 at budget 4, because zeroing a decoy costs nothing. Flips stay hard at budget 4: magnitude
  order 0.40–0.53, oracle 0.99, REINFORCE 0.92.
- **Ranking candidates** with a cheap learner (logistic regression on static features, no scoring): first pick right
  0.38 [0.25, 0.52] on decoy1 (chance 0.14) and 0.54 [0.43, 0.65] on decoy3 (chance 0.33), so the weights do carry a
  readable cue (the hidden unit's output weight and the input block). On flips it is 0.23 [0.12, 0.38] against 0.11
  for magnitude order — not separated. An LLM ranking run is worth a GPU try on decoys only, with the ranker as the bar.

## 6e. Teacher signal: the math top-up data (Kaggle T4×2, vLLM GPTQ-Int4, 2.0 h)

- 289 math questions answered with a 16,384-token limit (`cache/heal_traces/teacher2/traces.jsonl`): 261 finished,
  28 cut at the limit. Of the finished ones 253 are right by the benchmark's grader and 8 wrong; the right answers
  hold 1.42M completion tokens (median answer 5,529 tokens). vLLM gave 270 tokens/s against llama.cpp's 118.
- The run used about 2.4 of the 3 GPU-hours left this week, so **no training run fits before the weekly reset**. The
  top-20 probabilities and the recovery data were not made. Next training input: these 253 answers (wrong and cut
  ones dropped) on top of the rehearsal 5 mix.
