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

\section*{6. Training Process Summary}
\begin{enumerate}
    \item \textbf{Stage 1: Component Pre-training.} Pre-train G by discovering and imitating golden paths from the `kb.txt`. Concurrently or sequentially, pre-train D using positive samples from the `kb.txt` and generated negative samples.
    \item \textbf{Stage 2: Adversarial Training.} G and D are trained alternately. D provides reward signals to G, and G provides challenging negative samples for D.
    \item \textbf{Stage 3: Inference.} The trained Generator G is used with beam search to find the Top-K most probable reasoning paths and their terminal entities, which are returned as answers. The Discriminator is discarded.
\end{{document}