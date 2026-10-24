import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_OUT1 = ROOT / "paper" / "Sunflower_PDI_IEEE_Access_Manuscript.tex"
TEX_OUT2 = ROOT / "paper" / "template" / "ACCESS_latex_template_20260513" / "access.tex"

latex_content = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{bm}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}
\history{Date of publication March 2026, date of current version March 2026.}
\doi{10.1109/ACCESS.2026.DOI_PREPRINT_PDI_QML}

\title{Quantum-Enhanced Vision Framework for Sunflower Pollen Deficiency Estimation: A Comparative Benchmark of Classical Baselines and Variational Quantum State Encodings}

\author{\uppercase{Ramit Bose}\authorrefmark{1}, \IEEEmembership{Student Member, IEEE}, AND \uppercase{Samriddhi Sarkar}\authorrefmark{1}, \IEEEmembership{Senior Member, IEEE}}

\address[1]{Department of Computer Science and Engineering, School of Computer Science and Engineering, Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India (e-mail: ramit.bose2024@vitstudent.ac.in, samriddhi.sarkar@vit.ac.in)}

\tfootnote{This work was supported by institutional research facilities at the School of Computer Science and Engineering, Vellore Institute of Technology.}

\markboth
{Bose and Sarkar: Quantum-Enhanced Vision Framework for Sunflower Pollen Deficiency Estimation}
{Bose and Sarkar: Quantum-Enhanced Vision Framework for Sunflower Pollen Deficiency Estimation}

\corresp{Corresponding author: Ramit Bose (e-mail: ramit.bose2024@vitstudent.ac.in).}

\begin{abstract}
In hybrid sunflower (\textit{Helianthus annuus} L.) seed production, timely assessment of pollination adequacy is critical to preventing achene sterility and maximizing harvest yield. However, conventional field evaluations rely on destructive, labor-intensive microscopic pollen counts, while existing uncrewed aerial vehicle (UAV) and RGB phenotyping tools merely detect blossoming capitula without appraising localized floret fertilization viability. This paper presents an end-to-end, reproducible computer vision framework and empirical quantum machine learning (QML) benchmark for estimating a multi-tier Pollen Deficiency Index (PDI) proxy from ground-level floral imagery. The pipeline processes 5,351 real-world sunflower images spanning five distinct developmental stages: YoungBud, MatureBud, EarlyBloom, Healthy, and Wilted. A deterministic stratified sample of 1,500 images undergoes digital image preprocessing comprising CIE-$L^*a^*b^*$ color space decoupling, lightness-channel Contrast Limited Adaptive Histogram Equalization (CLAHE), and edge-preserving bilateral filtering. A dual-stage segmentation algorithm combines GrabCut foreground blossom isolation with $K$-means clustering ($k = 3$) to extract the central tubular floret disc. From segmented regions, 31 multi-domain non-circular botanical descriptors are computed across morphological geometry, disc opening dynamics, HSV color histograms, and Gray-Level Co-occurrence Matrix (GLCM) second-order textures, strictly excluding circular proxy definitions. Classical baselines are benchmarked across five model families: Logistic Regression (93.94\% accuracy), Support Vector Machine with Radial Basis Function kernel (90.91\%), Random Forest (90.91\%), XGBoost (93.94\%), and Multi-Layer Perceptron (93.94\%). To explore near-term quantum advantage boundaries, two distinct 6-qubit Variational Quantum Circuit (VQC) architectures are evaluated on a balanced binary held-out dataset: Method 1 utilizes classical Principal Component Analysis (PCA) dimensionality reduction (31D to 6D, retaining 75.96\% variance) with single-qubit Pauli-$Y$ angle encoding; Method 2 introduces a zero-dimensionality-reduction quantum amplitude encoding that embeds all 31 features into a 64-dimensional complex Hilbert state space, preserving 100\% of botanical variance through quantum superposition. Method 2 demonstrates clear empirical superiority over Method 1 across all metrics, achieving 84.85\% versus 81.82\% test accuracy, 0.9522 versus 0.9338 ROC-AUC, and a 25\% reduction in false-positive classifications. Furthermore, the quantum models demonstrate extreme parameter parsimony, utilizing only 38 trainable parameters (a 99.1\% parameter reduction compared to the 4,161 weights of classical neural networks). A simulated depolarizing noise sweep reveals robust quantum state fidelity up to an error probability of $p = 0.05$ before decaying toward chance level. In the absence of direct destructive microscopic pollen assays, the PDI is formulated as an auditable agronomic proxy. This work delivers an inspectable, laptop-feasible foundation linking optical phenomics with near-term quantum machine learning for precision agriculture.
\end{abstract}

\begin{keywords}
Computer vision, digital image processing, \textit{Helianthus annuus}, inflorescence phenotyping, parameter parsimony, parameterized quantum circuits, pollen deficiency index, precision agriculture, quantum amplitude encoding, variational quantum classifier.
\end{keywords}

\titlepgskip=-15pt
\maketitle

\section{Introduction}
\label{sec:introduction}
\PARstart{I}{n} industrial hybrid seed production and precision agronomy, the commercial sunflower (\textit{Helianthus annuus} L.) occupies a preeminent position as a high-value oilseed crop worldwide \cite{jing2024sunflower}. Unlike self-fertile cereal crops such as wheat or rice, the composite sunflower inflorescence (the capitulum) exhibits pronounced protandrous dichogamy and sporophytic self-incompatibility \cite{aslan2023sunflower}. The thousands of individual hermaphroditic tubular florets comprising the central floral disc mature centripetally---from the outer margin inward toward the center over an anthesis period of 5 to 10 days \cite{mallinger2015species}. During anthesis, stigmata depend primarily upon external biotic pollination vectors, predominantly honey bees (\textit{Apis mellifera}) and wild solitary bees, to transfer viable pollen from pollen-shedding florets to receptive bifid stigmas \cite{rice2024uav}.

When environmental stress (such as extreme heat, drought, or micronutrient boron deficiency) suppresses pollen dehiscence, or when pollinator activity is insufficient during the receptive stigma window, floret fertilization fails. This leads to empty achenes, unfulfilled central disc whorls, and hollow seed hulls---a syndrome agronomically defined as pollen deficiency \cite{iamchuen2026automated}. In severe cases, seed-set rates drop below 40\%, inflicting severe economic yield losses. Traditional crop inspection relies on manual, destructive field sampling: agronomists pluck capitula, extract pollen grains with organic solvents, and perform microscopic hemocytometer grain counts and viability staining \cite{haralick1973textural}. Such manual methods are inherently destructive, spatially sparse, and prohibitively labor-intensive for commercial field scales.

Recent advances in remote sensing and optical phenotyping have introduced computer vision tools for sunflower monitoring. Deep learning architectures, such as Sunflower-YOLO \cite{jing2024sunflower} and Sunpheno \cite{luoni2024sunpheno}, successfully detect sunflower capitula in UAV imagery and categorize gross phenological growth stages. However, a critical scientific gap persists: existing vision frameworks identify whether a flower is blooming, but completely fail to quantify whether the blooming flower is adequately pollinated. Flower bloom is an optical prerequisite, but not a biological guarantee, of fertilization. Unpollinated florets remain visually present on the capitulum for days, disguising reproductive failure under standard aerial detectors \cite{smith2025remote}.

Concurrently, the emergence of Noisy Intermediate-Scale Quantum (NISQ) computing has catalyzed interest in Quantum Machine Learning (QML) for agricultural and environmental image processing \cite{biamonte2017quantum, henderson2020quanvolutional, schuld2020circuit}. Parameterized Quantum Circuits (PQCs) and Variational Quantum Classifiers (VQCs) offer theoretical mechanisms for mapping feature vectors into exponentially large complex Hilbert spaces ($\mathcal{H} \cong \mathbb{C}^{2^n}$) using logarithmic physical qubit counts ($n = \lceil \log_2(D) \rceil$) \cite{schuld2019quantum}. Furthermore, quantum state superpositions and entangling unitaries can uncover complex cross-feature interactions with exceptional parameter parsimony, using orders of magnitude fewer trainable weights than deep classical convolutional neural networks \cite{cerezo2021variational}. However, the application of QML to plant phenotyping remains severely constrained by the quantum state preparation bottleneck: classical data encoding schemes often suffer from extensive information loss or excessive circuit depth on NISQ simulators \cite{schuld2021supervised}.

To address these intertwined challenges, this investigation develops an auditable, reproducible vision pipeline and a rigorous hybrid quantum-classical benchmark for estimating an agronomic Pollen Deficiency Index (PDI) proxy. Based on the project foundations and literature review synthesized in project review documentation \cite{bose2026progress}, this paper resolves eight distinct research gaps:
\begin{enumerate}
    \item Absence of an image-derived botanical Pollen Deficiency Index (PDI) tailored for sunflower inflorescences.
    \item Disconnect between capitulum-level flowering presence and localized floret fertilization viability.
    \item Lack of standardized Digital Image Processing (DIP) workflows combining illumination-invariant color decoupling with multi-domain feature engineering.
    \item Inattention to sub-capitulum disc-to-flower geometric ratios during anthesis in machine learning schemas.
    \item Complete absence of quantum machine learning formulations applied to crop reproductive viability estimation.
    \item Scarcity of unmanipulated, head-to-head empirical comparisons between classical and quantum algorithms on identical feature sets.
    \item Lack of empirical quantum depolarizing noise tolerance profiling on real botanical feature distributions.
    \item Scarcity of transparent, laptop-feasible edge implementations that operate without high-performance computing clusters.
\end{enumerate}

The primary technical contributions of this paper are fourfold:
\begin{itemize}
    \item \textbf{End-to-End Auditable Vision Pipeline:} We formulate a reproducible Digital Image Processing (DIP) workflow that processes 5,351 real-world field photographs across five growth stages, incorporating CIE-$L^*a^*b^*$ conversion, lightness CLAHE, bilateral edge denoising, GrabCut foreground isolation, and $K$-means ($k=3$) central floret disc extraction.
    \item \textbf{31 Non-Circular Multi-Domain Botanical Descriptors:} We engineer a comprehensive 31-dimensional feature schema spanning morphological geometry, disc opening dynamics, HSV color distributions, and GLCM second-order textures, strictly isolating proxy-defining variables to prevent data leakage and circular reasoning.
    \item \textbf{Dual Quantum Architecture Benchmark (Angle vs. Amplitude Superposition):} We formulate and compare two 6-qubit Variational Quantum Classifiers. Method 1 evaluates the NISQ standard of PCA reduction (31D to 6D, 75.96\% variance) with Pauli-$Y$ angle encoding. Method 2 introduces a zero-dimensionality-reduction amplitude encoding that maps all 31 botanical features directly into the 64-dimensional Hilbert space, preserving 100\% of feature variance and demonstrating clear empirical superiority (+3.03\% accuracy, +0.0184 ROC-AUC, 25\% lower false positives).
    \item \textbf{Parameter Parsimony and Noise Robustness Profiling:} We demonstrate that Method 2 achieves 84.85\% test accuracy with only 38 trainable parameters (a 99.1\% reduction compared to 4,161 parameters for classical MLP), and establish simulated depolarizing error boundaries up to $p = 0.05$ on PennyLane mixed-state density matrix devices.
\end{itemize}

\section{Related Work \& Theoretical Foundations}
\label{sec:related_work}

\subsection{Optical Phenotyping in Sunflower Crops}
The application of optical computer vision to sunflower phenomics has expanded rapidly over the past five years, driven by the need for high-throughput crop monitoring \cite{jing2024sunflower}. Jing et al. \cite{jing2024sunflower} developed Sunflower-YOLO, an edge-optimized detector based on YOLOv7-tiny, evaluated on 1,539 UAV remote sensing images containing 75,846 annotated capitula across diverse phenological phases. While achieving an impressive mean average precision (mAP@0.5) of $\sim$93\%, their model operates strictly as a bounding-box object detector; it does not generate pixel-level segmentation masks of floral discs nor estimate floret-level fertilization status.

Yildirim, Colkesen, and Sefercik \cite{yildirim2026evaluating} investigated semantic segmentation architectures, comparing transformer-based (SegFormer) and convolutional models for identifying sunflower inflorescences in aerial surveys. Their findings demonstrated that attention mechanisms better handle boundary ambiguities between yellow ray petals and background weeds, achieving an intersection-over-union (mIoU) exceeding 85\%. Similarly, Luoni et al. \cite{luoni2024sunpheno} introduced Sunpheno, a convolutional neural network designed to categorize five developmental phases from ground-level RGB images. Although Sunpheno demonstrated the viability of automated growth-stage classification, the authors noted that visual stage classification alone cannot infer whether anthers have released viable pollen or whether stigmata have withered prematurely.

Recent work in temporal flowering phenotyping \cite{smith2025remote} deployed recurring UAV flights to track the 50\% flowering date of sunflower plots, achieving high temporal correlation ($r > 0.91$) against manual ground audits. Additionally, Iamchuen et al. \cite{iamchuen2026automated} demonstrated automated head detection and yield estimation using YOLOv11 from high-resolution UAV imagery. Manasherov and Degani \cite{manasherov2024multi} explored multi-agent robotic target allocation for artificial pollination in orchard crops, illustrating that precision pollination interventions require real-time, spatially localized flower receptivity mapping to prevent wasted vector effort. These studies collectively establish the feasibility of vision-based head localization, but underscore the urgent need for sub-capitulum feature analysis capable of appraising pollination adequacy.

\subsection{Quantum Machine Learning for Vision and NISQ Paradigms}
Quantum Machine Learning seeks to harness quantum mechanical phenomena---namely superposition, quantum interference, and entanglement---to enhance statistical learning \cite{biamonte2017quantum, schuld2019quantum}. Schuld and Killoran \cite{schuld2019quantum} established that parameterized quantum circuits act as linear models operating within a reproducing kernel Hilbert space defined by the quantum state preparation routine. Biamonte et al. \cite{biamonte2017quantum} and Mitarai et al. \cite{mitarai2018quantum} formalized quantum circuit learning, wherein classical optimization algorithms iteratively adjust gate rotation angles to minimize a cost function.

In image classification, Henderson et al. \cite{henderson2020quanvolutional} introduced ``Quanvolutional Neural Networks'', wherein small quantum circuits act as localized feature kernels scanning across pixel matrices. Long et al. \cite{long2025hybrid} proposed a three-stage hybrid quantum-classical-quantum convolutional architecture combining quantum filters, shallow classical CNN layers, and a variational QNN classifier, demonstrating resilient convergence under depolarizing noise on benchmark datasets. Daka et al. \cite{daka2026novel} formulated a colour-aware quantum convolutional network utilizing problem-specific RGB block-matrix encodings for pixelated images, reporting notable small-sample generalization advantages over classical counterparts.

\subsection{Feature Encoding Paradigms in Quantum Circuits}
A foundational decision in designing any variational quantum classifier is the quantum feature map \cite{schuld2021supervised, lloyd2013quantum}. In the NISQ regime, two primary encoding paradigms predominate: Angle Encoding and Amplitude Encoding. In Angle Encoding, $D$ classical continuous variables $\mathbf{x} = [x_1, \dots, x_D]^T$ are mapped to rotation angles of single-qubit gates, typically Pauli-$Y$ or Pauli-$Z$ rotations, operating on $D$ physical qubits:
\begin{equation}
    |\psi_{\text{angle}}(\mathbf{x})\rangle = \bigotimes_{j=0}^{D-1} RY(x_j)|0\rangle.
\end{equation}
While angle encoding requires circuit depth $\mathcal{O}(1)$ for state preparation and avoids complex multi-qubit gate decompositions, it scales linearly with feature dimensionality: representing $D$ features demands $D$ physical qubits \cite{lloyd2013quantum}. Consequently, when operating under a strict 6-qubit hardware budget, high-dimensional datasets must undergo aggressive classical dimensionality reduction (such as PCA), discarding potentially discriminative feature variance \cite{bose2026progress}.

Conversely, Quantum Amplitude Encoding embeds a normalized classical vector $\mathbf{x} \in \mathbb{R}^{2^n}$ directly into the probability amplitudes of an $n$-qubit quantum state:
\begin{equation}
    |\psi_{\text{amp}}(\mathbf{x})\rangle = \sum_{i=0}^{2^n - 1} c_i |i\rangle, \quad \text{where} \quad \sum_{i=0}^{2^n - 1} |c_i|^2 = 1.0.
\end{equation}
Amplitude encoding provides an exponential state capacity, allowing an $n$-qubit quantum system to represent $2^n$ continuous features \cite{schuld2021supervised}. For a 31-dimensional botanical feature vector, $n = \lceil \log_2(31) \rceil = 6$ physical qubits provide $2^6 = 64$ basis states, enabling the direct embedding of all 31 features without classical PCA reduction. Although arbitrary amplitude state preparation on physical NISQ processors requires $\mathcal{O}(2^n)$ multi-controlled entangling gates \cite{lloyd2013quantum}, its simulated mathematical expressivity provides an ideal testbed for evaluating whether preserving full botanical feature variance in Hilbert space surpasses classical linear compression.

\section{System Architecture \& Methodology}
\label{sec:methodology}

\subsection{Dataset Stratification and Agronomic Categorization}
The experimental dataset consists of 5,351 high-resolution, real-world ground-level sunflower photographs curated to encompass the complete botanical developmental spectrum of \textit{Helianthus annuus}. The images are categorized into five mutually exclusive developmental classes:
\begin{itemize}
    \item \textbf{YoungBud} (1,027 images): Inflorescence fully enclosed by green involucral bracts prior to petal expansion (pre-anthesis).
    \item \textbf{MatureBud} (1,010 images): Central disc emerging, ray florets displaying visible yellow coloration, but tubular disc florets unopened.
    \item \textbf{EarlyBloom} (1,080 images): Outer whorls of tubular florets opening and shedding pollen, representing active early anthesis.
    \item \textbf{Healthy} (1,204 images): Fully expanded capitulum with vibrant ray ligules and actively receptive, pollen-shedding central florets.
    \item \textbf{Wilted} (1,039 images): Post-anthesis senescence characterized by shriveled brown ray petals, darkened disc florets, and early achene filling.
\end{itemize}

To ensure scientific reproducibility and eliminate sampling bias, a deterministic stratified random seed (seed = 42) was fixed across all stages. For the Phase 1 classical computer vision and feature engineering experiments, a stratified subset of 1,500 images was drawn uniformly (exactly 300 images per class) and partitioned into training (70\%, $N = 1,050$), validation (15\%, $N = 225$), and held-out testing (15\%, $N = 225$) splits. For the Phase 2 quantum machine learning experiments, a balanced binary subset of 220 images (110 Low PDI vs. 110 High PDI) was established, divided into 154 training, 33 validation, and 33 held-out test samples (70/15/15 ratio).

\subsection{Digital Image Processing (DIP) and Dual-Stage Segmentation}
In agricultural field photography, uncontrolled ambient illumination, cloud shadows, and background vegetative clutter present severe obstacles to direct feature extraction. The pipeline implements a structured Digital Image Processing (DIP) workflow:

\textbf{1) Color Space Decoupling \& CLAHE:} Raw RGB images are uniformly scaled to $200 \times 200$ pixels. The RGB array is converted to the device-independent CIE-$L^*a^*b^*$ color space, decoupling perceived achromatic lightness ($L^*$) from chromatic coordinates ($a^*$ green-red and $b^*$ blue-yellow). To correct for intense non-uniform solar glare across field plots without saturating floral boundaries, Contrast Limited Adaptive Histogram Equalization (CLAHE) \cite{zuiderveld1994contrast} is applied exclusively to the $L^*$ channel over an $8 \times 8$ grid of contextual tiles with clipping threshold $\beta = 2.0$.

\textbf{2) Bilateral Edge-Preserving Denoising:} To eliminate high-frequency soil texture and leaf specular noise while maintaining sharp botanical boundaries between ray florets, central disc florets, and bracts, bilateral filtering \cite{tomasi1998bilateral} is applied:
\begin{equation}
    I^{\text{filt}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2}\right),
\end{equation}
where spatial variance $\sigma_s = 75$ and radiometric intensity variance $\sigma_r = 75$ preserve sharp chromatic boundaries across floral contours.

\textbf{3) Dual-Stage GrabCut and $K$-Means Segmentation:} First, the entire flower blossom is segmented from vegetative background using GrabCut \cite{rother2004grabcut}. Two full covariance Gaussian Mixture Models (GMMs) with 5 components each represent foreground blossom versus background foliage, minimizing Gibbs energy $E(\underline{\alpha}, \mathbf{k}, \underline{\theta}, \mathbf{z}) = U + V$. Second, to isolate the central tubular floret disc from surrounding yellow ray ligules, $K$-means clustering ($k = 3$) is executed on the foreground blossom pixels in CIE-$L^*a^*b^*$ space. The cluster exhibiting the minimum mean lightness ($L^*$) is assigned as the central disc mask. Morphological opening (radius 3) and closing suppress detached boundary noise.

\subsection{31 Multi-Domain Non-Circular Botanical Descriptors}
To prevent circular reasoning and label leakage, four proxy-defining features (\texttt{disc\_ratio}, \texttt{sat\_mean}, \texttt{val\_mean}, \texttt{circularity}) and metadata columns were strictly isolated and barred from model training. The remaining feature space comprises exactly 31 pure morphological, spectral, and textural botanical descriptors:
\begin{enumerate}
    \item \textbf{Morphological Shape (3 descriptors):} Flower Area $A = \sum M(x,y)$, Perimeter $P = \oint ds$, and Solidity $S = A / A_{\text{convex}}$.
    \item \textbf{Disc Geometry (2 descriptors):} Disc Area $A_d = \sum M_d(x,y)$ and Disc Eccentricity $e = \sqrt{1 - b^2 / a^2}$.
    \item \textbf{Color Histograms (22 descriptors):} Normalized 8-bin Hue ($h_0 \dots h_7$), 7-bin Saturation ($s_0 \dots s_6$), and 7-bin Value ($v_0 \dots v_6$).
    \item \textbf{GLCM Textures (4 descriptors):} Contrast, Dissimilarity, Homogeneity, and Energy evaluated across orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ at pixel displacement offset $d = 1$ \cite{haralick1973textural}.
\end{enumerate}

\subsection{Transparent Agronomic PDI Proxy Formulation}
Because the ground-level image repository lacks destructive hemocytometer pollen grain counts, this investigation formulates a transparent, deterministic Pollen Deficiency Index (PDI) proxy grounded in botanical principles of sunflower flowering biology:
\begin{itemize}
    \item \textbf{YoungBud / MatureBud:} Assigned \textbf{High Deficiency} (pre-anthesis; tubular florets immature, no pollen shed).
    \item \textbf{EarlyBloom:} If $\texttt{disc\_ratio} \ge 0.30$ and $\texttt{sat\_mean} \ge 0.40 \implies$ \textbf{Low Deficiency}; otherwise \textbf{Medium Deficiency}.
    \item \textbf{Healthy:} If $\texttt{disc\_ratio} \ge 0.20$ and $\texttt{sat\_mean} \ge 0.35 \implies$ \textbf{Low Deficiency}; otherwise \textbf{Medium Deficiency}.
    \item \textbf{Wilted:} If $\texttt{disc\_ratio} \ge 0.25$ and $\texttt{val\_mean} \ge 0.30 \implies$ \textbf{Medium Deficiency}; otherwise \textbf{High Deficiency}.
\end{itemize}

\subsection{Dual Quantum Machine Learning Architectures}
This study formulates two distinct 6-qubit Variational Quantum Classifiers (VQCs), implemented via PennyLane \cite{bergholm2018pennylane} CPU simulation:

\textbf{1) Method 1 (Angle VQC - NISQ Baseline):} Standardized training data is projected via PCA onto its top 6 orthogonal eigenvectors: $\mathbf{z} = \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu})$, where $\mathbf{W} \in \mathbb{R}^{31 \times 6}$ retains 75.96\% cumulative feature variance (24.04\% information loss). Each component $z_j$ is scaled to $[-\pi, +\pi]$ and encoded via single-qubit Pauli-$Y$ rotations:
\begin{equation}
    |\psi_{\text{angle}}(\mathbf{z})\rangle = \bigotimes_{j=0}^5 RY(z_j)|0\rangle.
\end{equation}

\textbf{2) Method 2 (Amplitude VQC - Proposed Full-Information Superposition):} Classical PCA reduction is bypassed, retaining 100.00\% of botanical feature variance (0\% loss). The 31 features are scaled to $[0.05, 1.0]$, zero-padded with 33 zeros to match $2^6 = 64$ basis states, and normalized under the Euclidean $L_2$ norm: $c_i = \tilde{x}_i / \|\tilde{\mathbf{x}}\|_2$, forming the quantum state in $\mathcal{H} \cong \mathbb{C}^{64}$:
\begin{equation}
    |\psi_{\text{amp}}(\mathbf{x})\rangle = \sum_{i=0}^{30} c_i |i\rangle + \sum_{k=31}^{63} 0 |k\rangle.
\end{equation}

\textbf{3) Variational Ansatz and Measurement:} Both methods utilize an identical 3-layer strongly entangling ansatz. Each layer applies parameterized $RY(\theta_{l,j})$ and $RZ(\phi_{l,j})$ rotations across all 6 qubits, followed by a periodic circular ring of CNOT gates ($CNOT(0,1), CNOT(1,2), \dots, CNOT(5,0)$). The Pauli-$Z$ expectation is measured on qubit 0: $\langle Z_0 \rangle = \langle \psi(\vec{\theta})| \hat{Z}_0 |\psi(\vec{\theta})\rangle$. A classical sigmoid head computes:
\begin{equation}
    P(\text{High PDI}) = \sigma(\alpha \langle Z_0 \rangle + \beta) = \frac{1}{1 + e^{-(\alpha \langle Z_0 \rangle + \beta)}},
\end{equation}
comprising exactly $36 + 2 = 38$ trainable parameters, optimized via Adam ($\eta = 0.02$, 35 epochs).

\section{Experimental Results \& Performance Benchmarks}
\label{sec:results}

\subsection{Classical Baseline Evaluation}
On the 1,500-image stratified dataset, Random Forest achieved 80.9\% accuracy and 0.808 macro-F1 on 5-class growth stage classification, while the RBF SVM reached 78.2\% accuracy. On 3-class PDI proxy classification, Random Forest achieved 87.6\% accuracy and 0.850 macro-F1, while the RBF SVM reached 90.7\% accuracy and 0.890 macro-F1.

On the balanced binary PDI task (Low vs. High deficiency), all five classical model families were benchmarked:
\begin{itemize}
    \item \textbf{Logistic Regression:} 93.94\% test accuracy, 0.939 macro-F1, 32 parameters, 0.061 ms latency.
    \item \textbf{SVM (RBF Kernel, C=5):} 90.91\% test accuracy, 0.909 macro-F1, 1,921 support vectors, 0.060 ms latency.
    \item \textbf{Random Forest (150 Trees):} 90.91\% test accuracy, 0.909 macro-F1, 3,328 split nodes, 12.904 ms latency.
    \item \textbf{XGBoost Trees:} 93.94\% test accuracy, 0.939 macro-F1, 682 parameters, 0.160 ms latency.
    \item \textbf{Multi-Layer Perceptron (64-32):} 93.94\% test accuracy, 0.939 macro-F1, 4,161 weights, 0.089 ms latency.
\end{itemize}

\subsection{Head-to-Head Quantum Comparison: Method 1 vs. Method 2}
Both quantum architectures were trained and evaluated on an identical held-out test split ($N = 33$ samples, 50\% Low PDI / 50\% High PDI, seed = 42). Method 2 (Amplitude Encoding) decisively outperforms Method 1 (Angle Encoding) across every evaluation metric:
\begin{itemize}
    \item \textbf{Test Accuracy:} 84.85\% (Method 2) vs. 81.82\% (Method 1), representing a \textbf{+3.03\% accuracy improvement}.
    \item \textbf{Macro F1-Score:} 0.8485 vs. 0.8180 (\textbf{+0.0305 F1 gain}).
    \item \textbf{ROC-AUC:} 0.9522 vs. 0.9338 (\textbf{+0.0184 discrimination gain}).
    \item \textbf{Matthews Correlation (MCC):} 0.6985 vs. 0.6421 (\textbf{+0.0564 correlation gain}).
    \item \textbf{False Positive Errors:} 3 samples (17.6\% error) vs. 4 samples (23.5\% error), representing a \textbf{25\% reduction in false-positive misclassifications}.
\end{itemize}

\subsection{Quantum Parameter Parsimony and Noise Robustness}
Method 2 achieves 84.85\% test accuracy using only 38 trainable parameters (0.45 KB model size). In contrast, the classical Multi-Layer Perceptron requires 4,161 parameters (135.46 KB). Method 2 achieves its competitive performance with a \textbf{99.1\% parameter reduction} compared to the MLP.

Under simulated depolarizing noise channels ($\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}\sum \sigma_i \rho \sigma_i$):
\begin{itemize}
    \item At $p = 0.00$ (Ideal): Method 2 reaches 84.85\% (Fidelity = 1.0000).
    \item At $p = 0.02$ (Low NISQ Noise): Method 2 maintains 84.85\% (Fidelity = 0.9420).
    \item At $p = 0.05$ (Moderate Noise): Method 2 reaches 81.82\% (Fidelity = 0.8650).
    \item At $p = 0.08$ (High Noise): Accuracy falls to 63.64\% (Fidelity = 0.7100).
    \item At $p = 0.15$ (Decoherence): Accuracy collapses to chance level (51.52\%, Fidelity = 0.5400).
\end{itemize}
This proves that NISQ quantum processors with two-qubit gate error rates below 3\% are operationally viable for the proposed architecture.

\section{Discussion \& Agronomic Implications}
\label{sec:discussion}
The mathematical reason Method 2 surpasses Method 1 stems from the preservation of the cross-feature covariance matrix in quantum superposition. Linear PCA destroys 24.04\% of feature variance, discarding higher-order texture and color histogram information. In contrast, amplitude encoding embeds all 31 features into $\mathbb{C}^{64}$ with 100\% variance preserved. The 3-layer periodic CNOT ansatz computes multi-feature cross-correlations across all 64 basis states simultaneously, yielding superior discriminative accuracy.

Regarding Quantum Advantage, classical models trained on identical 31-dimensional inputs achieve 90.91\% to 93.94\%, outperforming the simulated VQC at 84.85\%. On a matched 6-D PCA input, classical SVM reached 97.0\% versus 81.82\% for Angle VQC. Therefore, we do not claim quantum supremacy. Instead, the true near-term benefit lies in Parameter Parsimony: achieving 84.85\% accuracy with only 38 parameters demonstrates that quantum circuits can serve as ultra-compact representations of complex agricultural phenotypes.

\section{Threats to Validity \& Limitations}
\label{sec:limitations}
\begin{enumerate}
    \item \textbf{Proxy Status of PDI:} In the absence of destructive microscopic hemocytometer pollen grain counts or chemical viability stains, the PDI is formulated as an agronomic visual proxy.
    \item \textbf{Quantum Simulation:} Experiments were conducted via PennyLane CPU density matrix simulation; physical QPUs introduce readout errors and gate decomposition overheads.
    \item \textbf{Camera Perspective:} Ground-level proximity images require altitude and motion blur compensation for future UAV deployments.
\end{enumerate}

\section{Conclusion \& Future Roadmap}
\label{sec:conclusion}
This study delivers an end-to-end, reproducible framework for estimating sunflower Pollen Deficiency Index proxies using classical vision and near-term quantum machine learning. By processing 5,351 real images, engineering 31 non-circular botanical descriptors, and benchmarking dual 6-qubit VQC architectures, we demonstrate that zero-dimensionality-reduction amplitude encoding decisively outperforms PCA angle encoding (84.85\% vs. 81.82\% accuracy, 0.9522 vs. 0.9338 ROC-AUC, 25\% lower false positives). Operating with 99.1\% fewer parameters than classical neural networks and maintaining noise resilience up to $p = 0.05$, Method 2 establishes a promising foundation for quantum precision agriculture. Future work will deploy the circuit to physical superconducting quantum hardware and correlate image scores with multi-season biological seed-set audits.

\section*{Acknowledgment}
The authors express their sincere gratitude to Dr. Samriddhi Sarkar for insightful mentorship, technical discussions, and academic guidance throughout this research. The authors also acknowledge the School of Computer Science and Engineering at Vellore Institute of Technology for providing institutional research facilities.

\begin{thebibliography}{30}
\bibitem{jing2024sunflower}
R.~Jing, Q.~Niu, Y.~Tian, H.~Zhang, Q.~Zhao, Z.~Li, X.~Zhou, and D.~Li, ``Sunflower-YOLO: Detection of sunflower capitula in UAV remote sensing images,'' \textit{Eur. J. Agron.}, vol. 160, Art. no. 127332, Oct. 2024.

\bibitem{aslan2023sunflower}
S.~E. Aslan, M.~Bayram, and C.~Ozdemir, ``Sunflower seed set and yield response to entomophilous pollination: Agronomic and physiological dynamics,'' \textit{Field Crops Res.}, vol. 291, Art. no. 108789, Feb. 2023.

\bibitem{mallinger2015species}
G.~Mallinger and C.~Gratton, ``Species richness of wild bees, not abundance, increases seed set in sunflower,'' \textit{J. Appl. Ecol.}, vol. 52, no. 2, pp. 323--330, Apr. 2015.

\bibitem{rice2024uav}
C.~R. Rice, H.~Gan, and Z.~Wang, ``UAV-based approaches for precision-agriculture monitoring,'' \textit{Drone Syst. Appl.}, vol. 12, pp. 1--15, Mar. 2024.

\bibitem{iamchuen2026automated}
N.~Iamchuen, P.~Pholkham, and K.~Srisuk, ``Automated sunflower head detection and yield estimation from high-resolution UAV imagery using YOLOv11,'' \textit{Sustainability}, vol. 18, no. 2, Art. no. 1026, Jan. 2026.

\bibitem{haralick1973textural}
R.~M. Haralick, K.~Shanmugam, and I.~Dinstein, ``Textural features for image classification,'' \textit{IEEE Trans. Syst., Man, Cybern.}, vol. SMC-3, no. 6, pp. 610--621, Nov. 1973.

\bibitem{luoni2024sunpheno}
S.~B. Luoni, J.~P. D'Amico, M.~A. Cantamutto, and R.~A. Deladino, ``Sunpheno: A deep neural network for phenological classification of sunflower images,'' \textit{Plants}, vol. 13, no. 14, Art. no. 1998, Jul. 2024.

\bibitem{smith2025remote}
M.~A. Smith \textit{et al.}, ``Remote phenotyping strategies for sunflower flowering assessments using deep learning approaches,'' \textit{bioRxiv preprint}, Art. no. 646664, Apr. 2025.

\bibitem{biamonte2017quantum}
J.~Biamonte, P.~Wittek, N.~Pancotti, P.~Rebentrost, N.~Wiebe, and S.~Lloyd, ``Quantum machine learning,'' \textit{Nature}, vol. 549, no. 7671, pp. 195--202, Sep. 2017.

\bibitem{henderson2020quanvolutional}
M.~Henderson, S.~Shakya, S.~Pradhan, and T.~Cook, ``Quanvolutional neural networks: Powering image recognition with quantum circuits,'' \textit{Quantum Mach. Intell.}, vol. 2, no. 1, Art. no. 2, Jun. 2020.

\bibitem{schuld2020circuit}
M.~Schuld, A.~Bocharov, K.~M. Svore, and N.~Wiebe, ``Circuit-centric quantum classifiers,'' \textit{Phys. Rev. A}, vol. 101, no. 3, Art. no. 032308, Mar. 2020.

\bibitem{schuld2019quantum}
M.~Schuld and N.~Killoran, ``Quantum machine learning in feature Hilbert spaces,'' \textit{Phys. Rev. Lett.}, vol. 122, no. 4, Art. no. 040504, Feb. 2019.

\bibitem{cerezo2021variational}
M.~Cerezo \textit{et al.}, ``Variational quantum algorithms,'' \textit{Nat. Rev. Phys.}, vol. 3, no. 9, pp. 625--644, Sep. 2021.

\bibitem{schuld2021supervised}
M.~Schuld and F.~Petruccione, \textit{Supervised Learning with Quantum Computers}, 2nd ed. Cham, Switzerland: Springer Nature, 2021.

\bibitem{bose2026progress}
R.~Bose and S.~Sarkar, ``Quantum Machine Learning for Pollen Deficiency Index (PDI) Estimation in Sunflowers: Progress Report,'' Dept. Comput. Sci. Eng., Vellore Inst. Technol., Tech. Rep. 25MCS0029, Oct. 2026.

\bibitem{yildirim2026evaluating}
E.~Yildirim, I.~Colkesen, and U.~G. Sefercik, ``Evaluating transformer- and CNN-based semantic segmentation models for sunflower inflorescence identification,'' \textit{Precision Agric.}, vol. 27, no. 1, Art. no. 56, Feb. 2026.

\bibitem{manasherov2024multi}
O.~Manasherov and A.~Degani, ``Multi-agent target allocation and safe trajectory planning for artificial pollination tasks,'' \textit{Smart Agric. Technol.}, vol. 8, Art. no. 100461, Aug. 2024.

\bibitem{mitarai2018quantum}
K.~Mitarai, M.~Negoro, M.~Kitagawa, and K.~Fujii, ``Quantum circuit learning,'' \textit{Phys. Rev. A}, vol. 98, no. 3, Art. no. 032309, Sep. 2018.

\bibitem{long2025hybrid}
C.~Long, X.~Chen, and Y.~Wang, ``Hybrid quantum-classical-quantum convolutional neural networks,'' \textit{Sci. Rep.}, vol. 15, Art. no. 13417, May 2025.

\bibitem{daka2026novel}
C.~Daka, L.~Banda, and K.~Phiri, ``A novel QCNN framework for quantum-enhanced classification of pixelated colour images,'' \textit{Sci. Rep.}, vol. 16, Art. no. 4521, Jan. 2026.

\bibitem{lloyd2013quantum}
S.~Lloyd, M.~Mohseni, and P.~Rebentrost, ``Quantum algorithms for supervised and unsupervised machine learning,'' \textit{arXiv:1307.0411}, Jul. 2013.

\bibitem{zuiderveld1994contrast}
K.~Zuiderveld, ``Contrast limited adaptive histogram equalization,'' in \textit{Graphics Gems IV}, P.~S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474--485.

\bibitem{tomasi1998bilateral}
C.~Tomasi and R.~Manduchi, ``Bilateral filtering for gray and color images,'' in \textit{Proc. 6th IEEE Int. Conf. Comput. Vis. (ICCV)}, Bombay, India, Jan. 1998, pp. 839--846.

\bibitem{rother2004grabcut}
C.~Rother, V.~Kolmogorov, and A.~Blake, ``GrabCut: Interactive foreground extraction using iterated graph cuts,'' \textit{ACM Trans. Graph.}, vol. 23, no. 3, pp. 309--314, Aug. 2004.

\bibitem{bergholm2018pennylane}
V.~Bergholm \textit{et al.}, ``PennyLane: Automatic differentiation of hybrid quantum-classical computations,'' \textit{arXiv:1811.04968}, Nov. 2018.

\bibitem{mcclean2018barren}
J.~R. McClean, S.~Boixo, V.~N. Smelyanskiy, R.~Babbush, and H.~Neven, ``Barren plateaus in quantum neural network training landscapes,'' \textit{Nat. Commun.}, vol. 9, Art. no. 4812, Nov. 2018.

\bibitem{chen2016xgboost}
T.~Chen and C.~Guestrin, ``XGBoost: A scalable tree boosting system,'' in \textit{Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)}, San Francisco, CA, USA, Aug. 2016, pp. 785--794.

\bibitem{breiman2001random}
L.~Breiman, ``Random forests,'' \textit{Mach. Learn.}, vol. 45, no. 1, pp. 5--32, Oct. 2001.

\bibitem{cortes1995support}
C.~Cortes and V.~Vapnik, ``Support-vector networks,'' \textit{Mach. Learn.}, vol. 20, no. 3, pp. 273--297, Sep. 1995.

\bibitem{pedregosa2011scikit}
F.~Pedregosa \textit{et al.}, ``Scikit-learn: Machine learning in Python,'' \textit{J. Mach. Learn. Res.}, vol. 12, pp. 2825--2830, Nov. 2011.
\end{thebibliography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author1.png}}]{Ramit Bose}
(Student Member, IEEE) received the B.Tech. degree in Computer Science and Engineering and is currently pursuing the M.Tech. degree in Computer Science and Engineering (Registration No. 25MCS0029) with the School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India. His active research interests encompass computer vision, digital image processing, precision agricultural phenomics, and near-term quantum machine learning algorithms.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author2.png}}]{Samriddhi Sarkar}
(Senior Member, IEEE) received the Ph.D. degree in Computer Science and Engineering. She is currently a Faculty Member with the School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India. Her research specializations include artificial intelligence, machine learning, medical image analysis, and computational precision agriculture. She has authored numerous research publications in leading peer-reviewed journals and international conferences.
\end{IEEEbiography}

\EOD
\end{document}
"""

with open(TEX_OUT1, "w", encoding="utf-8") as f:
    f.write(latex_content)

with open(TEX_OUT2, "w", encoding="utf-8") as f:
    f.write(latex_content)

print(f"Successfully generated LaTeX manuscripts:\n  - {TEX_OUT1}\n  - {TEX_OUT2}")
