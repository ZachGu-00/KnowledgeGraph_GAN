\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{\textbf{KG-GAN: A Generative Approach to KGQA on MetaQA Dataset}}
\author{Your Name}
\date{\today}
\maketitle

\section*{1. Core Idea: Paradigm Shift from ``Retrieval'' to ``Construction''}

The core idea remains unchanged. We redefine Knowledge Graph Question Answering (KGQA) as an active, generative task. The framework consists of two primary components engaged in adversarial training:
\begin{itemize}
    \item \textbf{Path Constructor (Generator G):} A reinforcement learning-based agent that actively constructs a reasoning path, or more broadly, an \textbf{Evidence Graphlet}.
    \item \textbf{Logic Appraiser (Discriminator D):} A multi-dimensional evaluation network that critiques the generated graphlets from logical, semantic, and structural perspectives.
\end{itemize}

\section*{2. Overall Architecture: Self-Contained Environment and Adversarial Game}

\subsection*{Environment Definition}
The MetaQA dataset provides a self-contained Knowledge Base (`kb.txt`), which serves as the complete environment for our agent. The entire KG is small enough to be loaded into memory.
\begin{itemize}
    \item \textbf{Knowledge Graph (KG):} The environment is constructed by loading all triples from the `kb.txt` file.
    \item \textbf{Action Space:} At any entity $e_t$, the agent's action space consists of all outgoing edges (relations) from $e_t$ in the loaded KG.
\end{itemize}

\subsection*{Evidence Graphlet Generation}
Generator G's objective is to construct an information-rich \textbf{Evidence Graphlet}. This includes the core reasoning path found in the KG and the first-hop neighbors of all entities along this path, providing rich context for the Discriminator.

\subsection*{Multi-Dimensional Logic Appraisal}
Discriminator D remains a multi-head critique network, evaluating the generated evidence graphlet based on its structural coherence, semantic relevance, and logical flow.

\section*{3. Generator (G): Path Constructor}

The Generator is a reinforcement learning agent whose policy is parameterized by a Graph Neural Network (GNN).

\subsection*{State ($S_t$)}
The state at timestep $t$ is a tuple $S_t = ( q_{\text{emb}}, h_0, P_t )$, where $q_{\text{emb}}$ is the question embedding, $h_0$ is the initial topic entity embedding (extracted from `[brackets]` in the question), and $P_t = (e_0, r_1, e_1, \dots, r_t, e_t)$ is the path constructed so far.

\subsection*{Strategy Network ($\pi(a_t | S_t)$)}
The policy network is implemented as a query-conditioned Graph Attention Network (GAT) that operates on the KG. It outputs a probability distribution over valid actions from the current entity $e_t$.

\subsection*{Reward Function}
During adversarial training, the reward $R_t$ is derived from the Discriminator D's multi-dimensional evaluation score. To encourage efficiency, a small step penalty is introduced: $R_t = D(G(S_t)) - \lambda_{\text{step}}$.

\section*{4. Discriminator (D): Logic Appraiser}

The Discriminator is a multi-head GNN-based network that learns to differentiate high-quality evidence graphlets from low-quality ones. Its architecture consists of a shared GNN encoder and three critique heads.

\section*{5. Efficient Pre-Training Strategy for MetaQA (Major Revision)}

This stage is critical for model warm-up. Since MetaQA does not provide an explicit `InferentialChain`, we must first discover the "golden path" from the provided `kb.txt`.

\subsection*{Generator Pre-training: Supervised Policy Learning (Behavior Cloning)}
The goal is to teach G the basic syntax of path construction by imitating the golden paths found within the `kb.txt`.

\begin{itemize}
    \item \textbf{Training Target:} For each question-answer pair, the ground-truth reasoning path(s) must be found by searching the KG.
    \item \textbf{Process for a given question:}
        \begin{enumerate}
            \item Parse the head entity $e_0$ from the question (e.g., `[Top Hat]`) and the answer entity $e_a$ from the answer.
            \item **Discover Golden Path:** Search the loaded KG to find the path(s) $P^*$ connecting $e_0$ to $e_a$. For MetaQA, this is often a 1-hop or 2-hop path. For example, find $r_1$ such that $(e_0, r_1, e_a)$ exists in the KG.
            \item \textbf{Imitate Path:} Use the discovered path $P^*$ as the ground-truth sequence for imitation learning. The training objective is to maximize the probability of selecting the correct relations from $P^*$ at each step, using a cross-entropy loss.
        \end{enumerate}
    \item \textbf{Handling Different Hop Counts:} The training curriculum is naturally formed by the length of the discovered golden paths ($|P^*|$).
\end{itemize}

\subsection*{Discriminator Pre-training: Learning Quality Metrics}
The goal is to pre-train D to recognize what constitutes a "good" evidence graphlet based on the provided `kb.txt`.

\begin{itemize}
    \item \textbf{Positive Samples ($g^+$):} The "golden evidence graphlets" are constructed from the true paths discovered in the KG for each question.
    \item \textbf{Negative Samples ($g^-$):} We generate a rich set of "damaged" graphlets by introducing targeted corruptions based on the KG.
        \begin{itemize}
            \item \textbf{Path Corruption:} For a correct path $(e_0, r_1, e_1)$, replace the correct relation $r_1$ with a random relation that also originates from $e_0$ but leads to a different entity.
            \item \textbf{Answer Corruption:} For a correct path, keep the path structure but replace the final answer entity $e_a$ with another random entity from the KG.
            \item \textbf{Incomplete Path:} For a multi-hop golden path, use a truncated version of the path as a negative sample.
        \end{itemize}
    \item \textbf{Loss Function:} D is trained using a multi-head margin-based ranking loss to score $g^+$ higher than $g^-$.
    $$
    \mathcal{L}_D = \sum_{h \in \{\text{struct}, \text{sem}, \text{logic}\}} \lambda_h \cdot \max(0, \text{margin} - \text{score}_h(g^+) + \text{score}_h(g^-))
    $$
\end{itemize}

\setcion{Important: Pretraining for discriminator using EPR}
遍历MetaQA三元组，生成ER-APs（e.g., "The Godfather -> director"）和RR-APs（e.g., "director <-> actor"）。正样本从标注路径提取（e.g., 1-3跳链上的模式）；
2. EPR训练（生成全局先验）
Bi-Encoder（召回APs，从语义角度
输入：问题q（e.g., "Who directed The Godfather?"）编码为Vq；模式p（e.g., "movie -> director"）编码为Vp。
训练：对比损失L = -log(sigmoid(Vq·Vp_pos)) + log(sigmoid(Vq·Vp_neg))，margin=0.2。In-batch负样本（batch内其他q的正AP作为负）。焦点语义对齐（cosine>0.7视为好）。
MetaQA适应：用1-3跳问题微调，epochs=15，确保AP Recall@30>95%（覆盖多跳模式）。
Cross-Encoder（排序EPs，从逻辑角度）：
输入：拼接q + EP（e.g., "Who directed... [SEP] movie -> director -> ?x"）。
训练：交叉熵L = -sum(y_log(p))，y=1为覆盖答案的EP。正EP：组合APs覆盖最多答案（e.g., 多导演路径）；负EP：100/正，焦点逻辑连贯（e.g., 路径完整性分数>0.8）。
MetaQA适应：用3跳问题强调多跳逻辑，epochs=10，确保EP Precision@5>80%。
输出：Top-10 EPs及其分数（e.g., 0.85），作为GAT边权重先验（语义分数初始化注意力，逻辑分数聚合路径）。
3. GAT训练（注入EPR先验，精炼注意力）
模型架构：GAT（2-3层，heads=4，hidden_dim=128），查询条件化（注意力a_ij = Attention(h_i, h_j, Vq, e_ij)，e_ij含EPR分数）。
先验注入：
语义注入：EPR分数（e.g., 0.85）作为边特征初始（concat到边嵌入）。
逻辑注入：EPR EP作为路径模板，初始化注意力权重（a_init = softmax(EPR_score)）。 
训练流程：
输入：宏观子图（节点嵌入：KG实体向量，如TransE；边嵌入：关系向量 + EPR先验）。
损失：L_ranking = max(0, S(P-) - S(P+) + 0.3)，S(P)=sum(log(a_i))沿路径聚合；+ λ·LKL(DEPR || DGAT)，λ=0.5（保持先验一致）。
正路径：MetaQA标注链（e.g., 3跳多答案）；负路径：硬负（前2跳同，后1跳错，模拟逻辑断裂）。
MetaQA适应：用3跳问题迭代，epochs=20，确保Path AUC>0.9（逻辑区分）和Semantic Recall>90%（语义覆盖）。动态阈值>0.8过滤路径。
端到端优化：冻结EPR后联合微调GAT，MMR重排答案（λ=0.5，确保多答案多样性）。
\end{{document}