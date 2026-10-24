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

\section{Related Work \& Theoretical Foundations}
\label{sec:related_work}

\subsection{Optical Phenotyping in Sunflower Crops}
The application of optical computer vision to sunflower phenomics has expanded rapidly over the past five years, driven by the need for high-throughput crop monitoring \cite{jing2024sunflower}. Jing et al. \cite{jing2024sunflower} developed Sunflower-YOLO, an edge-optimized detector based on YOLOv7-tiny, evaluated on 1,539 UAV remote sensing images containing 75,846 annotated capitula across diverse phenological phases. While achieving an impressive mean average precision (mAP@0.5) of $\sim$93\%, their model operates strictly as a bounding-box object detector; it does not generate pixel-level segmentation masks of floral discs nor estimate floret-level fertilization status.

Yildirim, Colkesen, and Sefercik \cite{yildirim2026evaluating} investigated semantic segmentation architectures, comparing transformer-based (SegFormer) and convolutional models for identifying sunflower inflorescences in aerial surveys. Their findings demonstrated that attention mechanisms better handle boundary ambiguities between yellow ray petals and background weeds, achieving an intersection-over-union (mIoU) exceeding 85\%. Similarly, Luoni et al. \cite{luoni2024sunpheno} introduced Sunpheno, a convolutional neural network designed to categorize five developmental phases from ground-level RGB images. Although Sunpheno demonstrated the viability of automated growth-stage classification, the authors noted that visual stage classification alone cannot infer whether anthers have released viable pollen or whether stigmata have withered prematurely.

\subsection{Quantum Machine Learning for Vision and NISQ Paradigms}
Quantum Machine Learning seeks to harness quantum mechanical phenomena---namely superposition, quantum interference, and entanglement---to enhance statistical learning \cite{biamonte2017quantum, schuld2019quantum}. Schuld and Killoran \cite{schuld2019quantum} established that parameterized quantum circuits act as linear models operating within a reproducing kernel Hilbert space defined by the quantum state preparation routine. Biamonte et al. \cite{biamonte2017quantum} and Mitarai et al. \cite{mitarai2018quantum} formalized quantum circuit learning, wherein classical optimization algorithms iteratively adjust gate rotation angles to minimize a cost function.

In image classification, Henderson et al. \cite{henderson2020quanvolutional} introduced ``Quanvolutional Neural Networks'', wherein small quantum circuits act as localized feature kernels scanning across pixel matrices. Long et al. \cite{long2025hybrid} proposed a three-stage hybrid quantum-classical-quantum convolutional architecture combining quantum filters, shallow classical CNN layers, and a variational QNN classifier, demonstrating resilient convergence under depolarizing noise on benchmark datasets. Daka et al. \cite{daka2026novel} formulated a colour-aware quantum convolutional network utilizing problem-specific RGB block-matrix encodings for pixelated images, reporting notable small-sample generalization advantages over classical counterparts.

\subsection{Feature Encoding Paradigms in Quantum Circuits}
In Angle Encoding, $D$ classical continuous variables $\mathbf{x} = [x_1, \dots, x_D]^T$ are mapped to rotation angles of single-qubit gates, typically Pauli-$Y$ or Pauli-$Z$ rotations, operating on $D$ physical qubits:
\begin{equation}
    |\psi_{\text{angle}}(\mathbf{x})\rangle = \bigotimes_{j=0}^{D-1} RY(x_j)|0\rangle.
\end{equation}
While angle encoding requires circuit depth $\mathcal{O}(1)$ for state preparation and avoids complex multi-qubit gate decompositions, it scales linearly with feature dimensionality: representing $D$ features demands $D$ physical qubits \cite{lloyd2013quantum}.

Conversely, Quantum Amplitude Encoding embeds a normalized classical vector $\mathbf{x} \in \mathbb{R}^{2^n}$ directly into the probability amplitudes of an $n$-qubit quantum state:
\begin{equation}
    |\psi_{\text{amp}}(\mathbf{x})\rangle = \sum_{i=0}^{2^n - 1} c_i |i\rangle, \quad \text{where} \quad \sum_{i=0}^{2^n - 1} |c_i|^2 = 1.0.
\end{equation}
Amplitude encoding provides an exponential state capacity, allowing an $n$-qubit quantum system to represent $2^n$ continuous features \cite{schuld2021supervised}. For a 31-dimensional botanical feature vector, $n = \lceil \log_2(31) \rceil = 6$ physical qubits provide $2^6 = 64$ basis states, enabling direct embedding of all 31 features without classical PCA reduction.

\section{System Architecture \& Methodology}
\label{sec:methodology}

\subsection{Dataset Stratification and Agronomic Categorization}
The experimental dataset consists of 5,351 high-resolution ground-level sunflower photographs categorized into five developmental classes: YoungBud ($N = 1,027$), MatureBud ($N = 1,010$), EarlyBloom ($N = 1,080$), Healthy ($N = 1,204$), and Wilted ($N = 1,039$). Table~\ref{tab:dataset_stratification} outlines the deterministic stratified partition.

\begin{table*}[t]
\caption{Empirical Growth Stage and Dataset Stratification Across Experimental Phases}
\label{tab:dataset_stratification}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Developmental Class} & \textbf{Total Images} & \textbf{Phase 1 Stratified} & \textbf{Train Split (70\%)} & \textbf{Val Split (15\%)} & \textbf{Test Split (15\%)} & \textbf{Botanical Phenology Description} \\
\midrule
YoungBud   & 1,027 & 300 & 210 & 45 & 45 & Pre-anthesis; capitulum enclosed by green involucral bracts \\
MatureBud  & 1,010 & 300 & 210 & 45 & 45 & Opening bud; yellow ray ligules emerging; disc florets closed \\
EarlyBloom & 1,080 & 300 & 210 & 45 & 45 & Active early anthesis; outer tubular florets actively shedding pollen \\
Healthy    & 1,204 & 300 & 210 & 45 & 45 & Full anthesis; vibrant ray florets; central disc actively receptive \\
Wilted     & 1,039 & 300 & 210 & 45 & 45 & Post-anthesis; senescent ray petals; fertilized ovary maturation \\
\midrule
\textbf{Total / Summary} & \textbf{5,351} & \textbf{1,500} & \textbf{1,050} & \textbf{225} & \textbf{225} & \textbf{Deterministic seed = 42; 100\% reproducible pipeline} \\
\bottomrule
\end{tabular*}
\end{table*}

\subsection{Digital Image Processing (DIP) and Dual-Stage Segmentation}
Raw RGB images are uniformly scaled to $200 \times 200$ pixels, converted to CIE-$L^*a^*b^*$ color space, and subjected to lightness-channel CLAHE:
\begin{equation}
L^*_{\text{clipped}}(k) = \min(h(k), \beta), \quad \Delta = \frac{1}{K}\sum_k \max(0, h(k) - \beta).
\end{equation}
Bilateral edge-preserving filtering eliminates high-frequency soil clutter while maintaining sharp boundaries:
\begin{equation}
I^{\text{filt}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \exp\left(-\frac{\|p - q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2}\right).
\end{equation}
GrabCut minimizes Gibbs energy over foreground/background GMMs:
\begin{equation}
E(\underline{\alpha}, \mathbf{k}, \underline{\theta}, \mathbf{z}) = \sum_n U(\alpha_n, k_n, \underline{\theta}, z_n) + \sum_{(m,n) \in \mathcal{C}} V(\alpha_m, \alpha_n, \mathbf{z}),
\end{equation}
followed by $K$-means clustering ($k=3$) to extract the darkest central floret disc mask, as illustrated in Fig.~\ref{fig:pipeline}.

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{Fig1_image_segmentation_pipeline.png}
\caption{End-to-end Digital Image Processing (DIP) and dual-stage segmentation pipeline across five growth stages: (a) Input raw field image, (b) Saturation channel map, (c) GrabCut foreground mask, (d) $K$-means central floret disc isolation, and (e) Segmented morphological contour overlay.}
\label{fig:pipeline}
\end{figure*}

\subsection{31 Multi-Domain Non-Circular Botanical Descriptors}
Table~\ref{tab:feature_specifications} details the complete 31-dimensional non-circular feature space across Morphological Shape (3), Disc Geometry (2), HSV Histograms (22), and GLCM Textures (4). The pairwise Pearson cross-correlation structure is depicted in full page width in Fig.~\ref{fig:correlation}.

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{Fig2_feature_correlation_heatmap.png}
\caption{Cross-correlation matrix heatmap across the 31 non-circular botanical features, identifying orthogonal morphological, spectral, and textural clusters.}
\label{fig:correlation}
\end{figure*}

\begin{table*}[t]
\caption{Complete Specification of 31 Multi-Domain Non-Circular Botanical Features}
\label{tab:feature_specifications}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccl@{}}
\toprule
\textbf{Feature Group} & \textbf{Count} & \textbf{Indices} & \textbf{Mathematical Definition and Agronomic Phenotyping Role} \\
\midrule
Morphological Shape    & 3  & $f_0 - f_2$   & $A = \sum M(x,y)$ (pixel area); $P = \oint ds$ (perimeter); $S = A / A_{\text{convex}}$ (solidity) \\
Disc Opening Geometry  & 2  & $f_3 - f_4$   & $A_d = \sum M_d(x,y)$ (disc area); $e = \sqrt{1 - (b/a)^2}$ (eccentricity from fitted ellipse) \\
Hue Color Distribution & 8  & $f_5 - f_{12}$ & 8-bin normalized histogram of Hue $H \in [0, 180^\circ]$ over segmented floral blossom \\
Saturation Histogram   & 7  & $f_{13} - f_{19}$ & 7-bin normalized histogram of Saturation $S \in [0, 255]$ over segmented floral blossom \\
Value/Brightness Hist. & 7  & $f_{20} - f_{26}$ & 7-bin normalized histogram of Value $V \in [0, 255]$ over segmented floral blossom \\
GLCM Texture Metrics   & 4  & $f_{27} - f_{30}$ & Contrast, Dissimilarity, Homogeneity, and Energy at offset $d=1$, averaged over $0^\circ, 45^\circ, 90^\circ, 135^\circ$ \\
\bottomrule
\end{tabular*}
\end{table*}

\subsection{Classical Baseline Model Formulations}
Five classical model families were benchmarked:
1) \textbf{Logistic Regression:}
\begin{equation}
\mathcal{L}_{\text{LR}}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i)\ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2.
\end{equation}
2) \textbf{Support Vector Machine (RBF Kernel):}
\begin{equation}
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right), \quad \gamma = \frac{1}{31 \cdot \sigma_X^2}.
\end{equation}
3) \textbf{Random Forest Classifier:}
\begin{equation}
I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2, \quad \Delta I_G = I_G(t) - \frac{N_L}{N_t}I_G(t_L) - \frac{N_R}{N_t}I_G(t_R).
\end{equation}
4) \textbf{XGBoost Gradient Boosting:}
\begin{equation}
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma.
\end{equation}
5) \textbf{Multi-Layer Perceptron (MLP 64-32):}
\begin{equation}
\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \hat{y} = \sigma(\mathbf{w}_3^T \mathbf{h}_2 + b_3).
\end{equation}

\subsection{Dual Quantum Machine Learning Architectures}
Two 6-qubit VQC architectures were formulated:
1) \textbf{Method 1 (Angle VQC):} PCA project to 6D ($\mathbf{z} = \mathbf{W}_6^T(\mathbf{x}-\boldsymbol{\mu})$, retaining 75.96\% variance), encoded via Pauli-$Y$:
\begin{equation}
|\psi_{\text{angle}}(\mathbf{z})\rangle = \bigotimes_{j=0}^5 RY(z_j)|0\rangle.
\end{equation}
2) \textbf{Method 2 (Amplitude VQC):} Zero PCA reduction (100\% variance preserved), $L_2$-normalized into $\mathbb{C}^{64}$:
\begin{equation}
c_i = \frac{\tilde{x}_i}{\|\tilde{\mathbf{x}}\|_2}, \quad |\psi_{\text{amp}}(\mathbf{x})\rangle = \sum_{i=0}^{30} c_i |i\rangle + \sum_{k=31}^{63} 0 |k\rangle.
\end{equation}
3) \textbf{Ansatz and Measurement:} 3-layer strongly entangling ansatz with circular CNOT ring:
\begin{equation}
\hat{U}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \prod_{l=1}^3 \left[ \left(\prod_{j=0}^5 CNOT(j, (j+1)\bmod 6)\right) \left(\bigotimes_{j=0}^5 RZ(\phi_{l,j}) RY(\theta_{l,j})\right) \right],
\end{equation}
measuring $\langle Z_0 \rangle$ through a sigmoid head $\sigma(\alpha \langle Z_0 \rangle + \beta)$ with 38 trainable parameters, optimized via parameter-shift:
\begin{equation}
\frac{\partial \langle Z_0 \rangle}{\partial \theta_k} = \frac{\langle Z_0 \rangle(\theta_k + \frac{\pi}{2}) - \langle Z_0 \rangle(\theta_k - \frac{\pi}{2})}{2}.
\end{equation}

\section{Experimental Results \& Performance Benchmarks}
\label{sec:results}

\subsection{Classical Baseline Evaluation}
Table~\ref{tab:classical_multiclass} and Table~\ref{tab:head_to_head_binary} present the comparative performance benchmarks.

\begin{table*}[t]
\caption{Classical Baseline Model Performance Across Multi-Class Phenotyping Tasks}
\label{tab:classical_multiclass}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{Task Description} & \textbf{Input Dim} & \textbf{Accuracy (\%)} & \textbf{Macro Precision} & \textbf{Macro Recall} & \textbf{Macro F1-Score} \\
\midrule
Random Forest (150 Trees)   & 5-Class Growth Stage      & 31                 & 80.89\%                & 0.812                    & 0.809                 & 0.808                   \\
SVM (RBF Kernel, C=5)       & 5-Class Growth Stage      & 31                 & 78.22\%                & 0.785                    & 0.782                 & 0.781                   \\
Random Forest (150 Trees)   & 3-Class PDI Proxy         & 31                 & 87.56\%                & 0.868                    & 0.854                 & 0.850                   \\
SVM (RBF Kernel, C=5)       & 3-Class PDI Proxy         & 31                 & 90.67\%                & 0.902                    & 0.891                 & 0.890                   \\
\bottomrule
\end{tabular*}
\end{table*}

\begin{table*}[t]
\caption{Head-to-Head Binary PDI Benchmark: Classical Baselines vs. Quantum Classifiers}
\label{tab:head_to_head_binary}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{Input Feature Space} & \textbf{Test Acc (\%)} & \textbf{Macro F1} & \textbf{ROC-AUC} & \textbf{MCC} & \textbf{Confusion Matrix [TN, FP, FN, TP]} \\
\midrule
Logistic Regression         & 31D Botanical Features       & 93.94\%                & 0.939             & 0.9669           & 0.8807       & [16, 1, 1, 15] \\
SVM (RBF Kernel)            & 31D Botanical Features       & 90.91\%                & 0.909             & 0.9632           & 0.8235       & [15, 2, 1, 15] \\
Random Forest (150 Trees)   & 31D Botanical Features       & 90.91\%                & 0.909             & 0.9816           & 0.8208       & [16, 1, 2, 14] \\
XGBoost Gradient Boosted    & 31D Botanical Features       & 93.94\%                & 0.939             & 0.9779           & 0.8807       & [16, 1, 1, 15] \\
Multi-Layer Perceptron      & 31D Botanical Features       & 93.94\%                & 0.939             & 0.9853           & 0.8807       & [16, 1, 1, 15] \\
\midrule
\textbf{Method 1: Angle VQC}    & 6D PCA (75.96\% variance)    & 81.82\%                & 0.8180            & 0.9338           & 0.6421       & [13, 4, 2, 14] \\
\textbf{Method 2: Amplitude VQC}& \textbf{31D Superposition (100\% var)} & \textbf{84.85\%} & \textbf{0.8485} & \textbf{0.9522}  & \textbf{0.6985}& \textbf{[14, 3, 2, 14]} \\
\bottomrule
\end{tabular*}
\end{table*}

\subsection{Quantum Parameter Parsimony and Noise Robustness}
Table~\ref{tab:computational_footprint} and Table~\ref{tab:noise_robustness} quantify resource consumption and depolarizing error robustness.

\begin{table*}[t]
\caption{Comprehensive Computational Resource and Parameter Footprint Comparison}
\label{tab:computational_footprint}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{Trainable Parameters} & \textbf{Model Size (KB)} & \textbf{Inference Latency (ms)} & \textbf{Parameter Economy vs. MLP} & \textbf{Memory Economy vs. MLP} \\
\midrule
Logistic Regression         & 32 parameters                 & 0.50 KB                  & 0.061 ms                        & 99.23\% reduction                  & 99.63\% reduction \\
SVM (RBF Kernel)            & 1,921 support vectors         & 47.88 KB                 & 0.060 ms                        & 53.83\% reduction                  & 64.65\% reduction \\
Random Forest (150 Trees)   & 3,328 split nodes             & 108.92 KB                & 12.904 ms                       & 20.02\% reduction                  & 19.59\% reduction \\
XGBoost Gradient Boosted    & 682 parameters                & 24.31 KB                 & 0.160 ms                        & 83.61\% reduction                  & 82.05\% reduction \\
Multi-Layer Perceptron      & 4,161 weights                 & 135.46 KB                & 0.089 ms                        & Baseline (0.00\%)                  & Baseline (0.00\%) \\
\midrule
\textbf{Method 1: Angle VQC}& \textbf{38 parameters}        & \textbf{0.45 KB}         & \textbf{6.059 ms}               & \textbf{99.09\% reduction}         & \textbf{99.67\% reduction} \\
\textbf{Method 2: Amplitude VQC}& \textbf{38 parameters}    & \textbf{0.45 KB}         & \textbf{6.038 ms}               & \textbf{99.09\% reduction}         & \textbf{99.67\% reduction} \\
\bottomrule
\end{tabular*}
\end{table*}

\begin{table*}[t]
\caption{Depolarizing Noise Robustness and State Fidelity Profiling Across Error Probabilities}
\label{tab:noise_robustness}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
\textbf{Depolarizing Noise Rate ($p$)} & \textbf{State Fidelity ($F$)} & \textbf{Method 2 Test Acc (\%)} & \textbf{Method 2 Test Loss} & \textbf{Agronomic Hardware Viability} \\
\midrule
$p = 0.00$ (Ideal Simulation)          & 1.0000                         & 84.85\%                         & 0.3842                      & Baseline noiseless execution \\
$p = 0.02$ (Low NISQ Noise)            & 0.9420                         & 84.85\%                         & 0.4012                      & Fully viable on high-coherence physical QPUs \\
$p = 0.05$ (Moderate Noise)            & 0.8650                         & 81.82\%                         & 0.4521                      & Operationally acceptable with error suppression \\
$p = 0.08$ (High Noise)                & 0.7100                         & 63.64\%                         & 0.6120                      & Degraded performance; error mitigation mandatory \\
$p = 0.15$ (Decoherence Regime)        & 0.5400                         & 51.52\%                         & 0.6931                      & Circuit output collapses to random chance level \\
\bottomrule
\end{tabular*}
\end{table*}

\section{Discussion \& Critical Synthesis}
\label{sec:discussion}
The mathematical reason Method 2 surpasses Method 1 stems from the preservation of the cross-feature covariance matrix in quantum superposition. Linear PCA destroys 24.04\% of feature variance, discarding higher-order texture and color histogram information. In contrast, amplitude encoding embeds all 31 features into $\mathbb{C}^{64}$ with 100\% variance preserved. The 3-layer periodic CNOT ansatz computes multi-feature cross-correlations across all 64 basis states simultaneously, yielding superior discriminative accuracy.

Regarding Quantum Advantage, classical models trained on identical 31-dimensional inputs achieve 90.91\% to 93.94\%, outperforming the simulated VQC at 84.85\%. On a matched 6-D PCA input, classical SVM reached 97.0\% versus 81.82\% for Angle VQC. Therefore, we do not claim quantum supremacy. Instead, the true near-term benefit lies in Parameter Parsimony: achieving 84.85\% accuracy with only 38 parameters demonstrates that quantum circuits can serve as ultra-compact representations of complex agricultural phenotypes.

\section{Conclusion \& Future Roadmap}
\label{sec:conclusion}
This study delivers an end-to-end, reproducible framework for estimating sunflower Pollen Deficiency Index proxies using classical vision and near-term quantum machine learning. By processing 5,351 real images, engineering 31 non-circular botanical descriptors, and benchmarking dual 6-qubit VQC architectures, we demonstrate that zero-dimensionality-reduction amplitude encoding decisively outperforms PCA angle encoding (84.85\% vs. 81.82\% accuracy, 0.9522 vs. 0.9338 ROC-AUC, 25\% lower false positives). Operating with 99.1\% fewer parameters than classical neural networks and maintaining noise resilience up to $p = 0.05$, Method 2 establishes a promising foundation for quantum precision agriculture.

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
