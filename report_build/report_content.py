# -*- coding: utf-8 -*-
"""Report body text.

Third-person observational voice throughout, per the template's writing-style
note: no "I", "we", "my", "our" or "us" anywhere in this file.

Citations are written as ``[key]`` where ``key`` is a key of CITATIONS below.
``build_report.py`` resolves them to IEEE numbers in order of first appearance
and emits the reference list in that order, so nothing needs renumbering by
hand when the text is edited. Several keys may share one marker:
``[groh2021, wen2022]``.
"""

# --- title page ------------------------------------------------------------

TITLE = ("Does Bias Propagate? Fairness in Automated Acne Severity Assessment "
         "and Downstream Skincare Recommendation")
AUTHOR = "Sowmiya"
SUBTITLE = ("A two-stage pipeline audit combining disaggregated fairness "
            "measurement with a causal skin-tone probe")


# --- references ------------------------------------------------------------
# IEEE style. Numbering is assigned automatically at build time.

CITATIONS = {

 # --- fairness in dermatological and facial AI ---
 "esteva2017": 'A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau and S. Thrun, "Dermatologist-level classification of skin cancer with deep neural networks," Nature, vol. 542, no. 7639, pp. 115-118, 2017.',
 "groh2021": 'M. Groh, C. Harris, L. Soenksen, F. Lau, R. Han, A. Kim, A. Koochek and O. Badri, "Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2021, pp. 1820-1828.',
 "groh2022": 'M. Groh, C. Harris, R. Daneshjou, O. Badri and A. Koochek, "Towards transparency in dermatology image datasets with skin tone annotations by experts, crowds, and an algorithm," Proc. ACM on Human-Computer Interaction, vol. 6, no. CSCW2, art. 526, 2022.',
 "daneshjou2022": 'R. Daneshjou, K. Vodrahalli, R. A. Novoa, M. Jenkins, W. Liang, V. Rotemberg, J. Ko, S. M. Swetter, E. E. Bailey, O. Gevaert, P. Mukherjee, M. Phung, K. Yekrang, B. Fong, R. Sahasrabudhe, J. A. C. Allerup, U. Okata-Karigane, J. Zou and A. S. Chiou, "Disparities in dermatology AI performance on a diverse, curated clinical image set," Science Advances, vol. 8, no. 32, art. eabq6147, 2022.',
 "adamson2018": 'A. S. Adamson and A. Smith, "Machine learning and health care disparities in dermatology," JAMA Dermatology, vol. 154, no. 11, pp. 1247-1248, 2018.',
 "kinyanjui2020": 'N. M. Kinyanjui, T. Odonga, C. Cintas, N. C. F. Codella, R. Panda, P. Sattigeri and K. R. Varshney, "Fairness of classifiers across skin tones in dermatology," in Proc. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2020, pp. 320-329.',
 "bevan2022": 'P. J. Bevan and A. Atapour-Abarghouei, "Detecting melanoma fairly: Skin tone detection and debiasing for skin lesion classification," in MICCAI Workshop on Domain Adaptation and Representation Transfer, 2022, pp. 1-11.',
 "wen2022": 'D. Wen, S. M. Khan, A. J. Xu, H. Ibrahim, L. Smith, J. Caballero, L. Zepeda, C. de Blas Perez, A. K. Denniston, X. Liu and R. N. Matin, "Characteristics of publicly available skin cancer image datasets: A systematic review," The Lancet Digital Health, vol. 4, no. 1, pp. e64-e74, 2022.',
 "buolamwini2018": 'J. Buolamwini and T. Gebru, "Gender shades: Intersectional accuracy disparities in commercial gender classification," in Proc. Conf. Fairness, Accountability and Transparency (FAT*), PMLR vol. 81, 2018, pp. 77-91.',

 # --- skin tone measurement ---
 "fitzpatrick1988": 'T. B. Fitzpatrick, "The validity and practicality of sun-reactive skin types I through VI," Archives of Dermatology, vol. 124, no. 6, pp. 869-871, 1988.',
 "chardon1991": 'A. Chardon, I. Cretois and C. Hourseau, "Skin colour typology and suntanning pathways," International Journal of Cosmetic Science, vol. 13, no. 4, pp. 191-208, 1991.',
 "delbino2013": 'S. Del Bino and F. Bernerd, "Variations in skin colour and the biological consequences of ultraviolet radiation exposure," British Journal of Dermatology, vol. 169, suppl. 3, pp. 33-40, 2013.',
 "merler2019": 'M. Merler, N. Ratha, R. S. Feris and J. R. Smith, "Diversity in faces," arXiv preprint arXiv:1901.10436, 2019.',

 # --- fairness theory, composition and pipelines ---
 "dwork2012": 'C. Dwork, M. Hardt, T. Pitassi, O. Reingold and R. Zemel, "Fairness through awareness," in Proc. 3rd Innovations in Theoretical Computer Science Conf. (ITCS), 2012, pp. 214-226.',
 "hardt2016": 'M. Hardt, E. Price and N. Srebro, "Equality of opportunity in supervised learning," in Advances in Neural Information Processing Systems (NIPS), vol. 29, 2016, pp. 3315-3323.',
 "dwork2019": 'C. Dwork and C. Ilvento, "Fairness under composition," in Proc. 10th Innovations in Theoretical Computer Science Conf. (ITCS), 2019, art. 33, pp. 1-20.',
 "bower2017": 'A. Bower, S. N. Kitchen, L. Niss, M. J. Strauss, A. Vargas and S. Venkatasubramanian, "Fair pipelines," arXiv preprint arXiv:1707.00391, 2017.',
 "kusner2017": 'M. J. Kusner, J. R. Loftus, C. Russell and R. Silva, "Counterfactual fairness," in Advances in Neural Information Processing Systems (NIPS), vol. 30, 2017, pp. 4066-4076.',
 "mehrabi2021": 'N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman and A. Galstyan, "A survey on bias and fairness in machine learning," ACM Computing Surveys, vol. 54, no. 6, art. 115, 2021.',
 "barocas2023": 'S. Barocas, M. Hardt and A. Narayanan, Fairness and Machine Learning: Limitations and Opportunities. Cambridge, MA: MIT Press, 2023.',
 "corbett2018": 'S. Corbett-Davies and S. Goel, "The measure and mismeasure of fairness: A critical review of fair machine learning," arXiv preprint arXiv:1808.00023, 2018.',
 "selbst2019": 'A. D. Selbst, D. Boyd, S. A. Friedler, S. Venkatasubramanian and J. Vertesi, "Fairness and abstraction in sociotechnical systems," in Proc. Conf. Fairness, Accountability and Transparency (FAT*), 2019, pp. 59-68.',
 "sagawa2020": 'S. Sagawa, P. W. Koh, T. B. Hashimoto and P. Liang, "Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization," in Proc. International Conf. Learning Representations (ICLR), 2020.',

 # --- recommendation ---
 "lee2024": 'J. Lee, H. Yoon, S. Kim, C. Lee, J. Lee and S. Yoo, "Deep learning-based skin care product recommendation: A focus on cosmetic ingredient analysis and facial skin conditions," Journal of Cosmetic Dermatology, vol. 23, no. 6, pp. 2066-2077, 2024.',
 "ekstrand2022": 'M. D. Ekstrand, A. Das, R. Burke and F. Diaz, "Fairness in information access systems," Foundations and Trends in Information Retrieval, vol. 16, no. 1-2, pp. 1-177, 2022.',
 "wang2022": 'L. Wang and T. Joachims, "Fairness in the first stage of two-stage recommender systems," arXiv preprint arXiv:2205.15436, 2022.',
 "abdollahpouri2019": 'H. Abdollahpouri, R. Burke and B. Mobasher, "Managing popularity bias in recommender systems with personalized re-ranking," in Proc. 32nd International Florida Artificial Intelligence Research Society Conf. (FLAIRS), 2019, pp. 413-418.',
 "ge2010": 'M. Ge, C. Delgado-Battenfeld and D. Jannach, "Beyond accuracy: Evaluating recommender systems by coverage and serendipity," in Proc. 4th ACM Conf. Recommender Systems (RecSys), 2010, pp. 257-260.',

 # --- architectures and training ---
 "he2016": 'K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.',
 "tan2019": 'M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. 36th International Conf. Machine Learning (ICML), PMLR vol. 97, 2019, pp. 6105-6114.',
 "dosovitskiy2021": 'A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. International Conf. Learning Representations (ICLR), 2021.',
 "deng2009": 'J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248-255.',
 "lin2017": 'T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar, "Focal loss for dense object detection," in Proc. IEEE International Conf. Computer Vision (ICCV), 2017, pp. 2980-2988.',
 "vaswani2017": 'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin, "Attention is all you need," in Advances in Neural Information Processing Systems (NIPS), vol. 30, 2017, pp. 5998-6008.',
 "devlin2019": 'J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. Conf. North American Chapter of the ACL (NAACL-HLT), 2019, pp. 4171-4186.',
 "sanh2019": 'V. Sanh, L. Debut, J. Chaumond and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," arXiv preprint arXiv:1910.01108, 2019.',
 "wu2019": 'X. Wu, N. Wen, J. Liang, Y.-K. Lai, D. She, M.-M. Cheng and J. Yang, "Joint acne image grading and counting via label distribution learning," in Proc. IEEE/CVF International Conf. Computer Vision (ICCV), 2019, pp. 10642-10651.',

 # --- calibration, abstention and statistical practice ---
 "guo2017": 'C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. 34th International Conf. Machine Learning (ICML), PMLR vol. 70, 2017, pp. 1321-1330.',
 "naeini2015": 'M. P. Naeini, G. F. Cooper and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in Proc. 29th AAAI Conf. Artificial Intelligence, 2015, pp. 2901-2907.',
 "geifman2017": 'Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in Advances in Neural Information Processing Systems (NIPS), vol. 30, 2017, pp. 4878-4887.',
 "elyaniv2010": 'R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," Journal of Machine Learning Research, vol. 11, pp. 1605-1641, 2010.',
 "efron1993": 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman and Hall/CRC, 1993.',
 "cohen1988": 'J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988.',
 "bouthillier2021": 'X. Bouthillier, P. Delaunay, M. Bronzi, A. Trofimov, B. Nichyporuk, J. Szeto, N. Mohammadi Sepahvand, E. Raff, K. Madan, V. Voleti, S. Ebrahimi Kahou, V. Michalski, T. Arbel, C. Pal, G. Varoquaux and P. Vincent, "Accounting for variance in machine learning benchmarks," in Proc. Machine Learning and Systems (MLSys), vol. 3, 2021, pp. 747-769.',
 "picard2021": 'D. Picard, "torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision," arXiv preprint arXiv:2109.08203, 2021.',

 # --- clinical and dermatological grounding ---
 "hayashi2008": 'N. Hayashi, H. Akamatsu and M. Kawashima, "Establishment of grading criteria for acne severity," Journal of Dermatology, vol. 35, no. 5, pp. 255-260, 2008.',
 "zaenglein2016": 'A. L. Zaenglein, A. L. Pathy, B. J. Schlosser, A. Alikhan, H. E. Baldwin, D. S. Berson, W. P. Bowe, E. M. Graber, J. C. Harper, S. Kang, J. E. Keri, J. J. Leyden, R. C. Reynolds, N. B. Silverberg, L. F. Stein Gold, M. M. Tollefson, J. W. Weiss, N. C. Dolan, A. A. Sagan, M. Stern, K. M. Boyer and R. Bhushan, "Guidelines of care for the management of acne vulgaris," Journal of the American Academy of Dermatology, vol. 74, no. 5, pp. 945-973, 2016.',
 "davis2010": 'E. C. Davis and V. D. Callender, "Postinflammatory hyperpigmentation: A review of the epidemiology, clinical features, and treatment options in skin of color," Journal of Clinical and Aesthetic Dermatology, vol. 3, no. 7, pp. 20-31, 2010.',
 "bissett2005": 'D. L. Bissett, J. E. Oblong and C. A. Berge, "Niacinamide: A B vitamin that improves aging facial skin appearance," Dermatologic Surgery, vol. 31, no. 7, pp. 860-865, 2005.',
 "arif2015": 'T. Arif, "Salicylic acid as a peeling agent: A comprehensive review," Clinical, Cosmetic and Investigational Dermatology, vol. 8, pp. 455-461, 2015.',

 # --- documentation practice ---
 "gebru2021": 'T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daume III and K. Crawford, "Datasheets for datasets," Communications of the ACM, vol. 64, no. 12, pp. 86-92, 2021.',
 "mitchell2019": 'M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji and T. Gebru, "Model cards for model reporting," in Proc. Conf. Fairness, Accountability and Transparency (FAT*), 2019, pp. 220-229.',

 # --- software and data ---
 "paszke2019": 'A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai and S. Chintala, "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 8024-8035.',
 "wightman2019": 'R. Wightman, "PyTorch image models," GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models',
 "wolf2020": 'T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest and A. M. Rush, "Transformers: State-of-the-art natural language processing," in Proc. Conf. Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP), 2020, pp. 38-45.',
 "pedregosa2011": 'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot and E. Duchesnay, "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',
 "harris2020": 'C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del Rio, M. Wiebe, P. Peterson, P. Gerard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke and T. E. Oliphant, "Array programming with NumPy," Nature, vol. 585, no. 7825, pp. 357-362, 2020.',
 "mckinney2010": 'W. McKinney, "Data structures for statistical computing in Python," in Proc. 9th Python in Science Conf. (SciPy), 2010, pp. 56-61.',
 "bradski2000": 'G. Bradski, "The OpenCV library," Dr. Dobb\'s Journal of Software Tools, vol. 25, no. 11, pp. 120-125, 2000.',
 "streamlit2019": 'Streamlit Inc., "Streamlit: A faster way to build and share data apps," 2019. [Online]. Available: https://streamlit.io',
 "sephora2023": 'Nadyinky, "Sephora products and skincare reviews," Kaggle dataset, 2023. [Online]. Available: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews',
}


# --- front matter ----------------------------------------------------------

ACKNOWLEDGEMENTS = [
 "The author would like to thank her supervisor for guidance throughout this project, and in particular for the insistence on validating the skin tone estimator against real images before building on it. That requirement located a measurement error which would otherwise have propagated through every subsequent analysis, and the correction it forced is described in the reflection of Chapter 5.",
 "Thanks are also due to the teaching staff of the department for their instruction over the course of the programme, and to the author's family and friends for their patience and support while this work was carried out.",
]

ABSTRACT = [
 "AI-driven skin analysis has moved rapidly from research into consumer products, with major retailers now offering tools that photograph a user's face, assess skin concerns, and recommend products accordingly. Such systems are trained on image corpora that under-represent darker skin tones, and the resulting harm is realised not at the point of classification but at the point of recommendation: a misassessed user is shown a different set of products. Fairness auditing of dermatological models is well established but stops at the classifier and addresses diagnosis; fairness across composed pipelines is well developed but situated in employment, credit and content ranking. Neither has been applied to consumer skincare recommendation.",

 "This project occupies that intersection. A two-stage pipeline was constructed, comprising a ResNet-50 acne severity classifier trained on 1,457 dermatologist-graded facial images and a deterministic content-based recommender over a catalogue of 2,420 skincare products and 1.09 million reviews, connected by a documented severity-to-ingredient mapping. Fairness was measured at both stages and across the boundary, with bootstrapped confidence intervals and effect sizes throughout, under a protocol fixed before results were inspected.",

 "Skin tone was estimated using the Individual Typology Angle, and the first implementation failed: it assigned 59.5% of a predominantly Chinese corpus to the darkest band. Visual validation identified two confounds, shadow and acne erythema, the latter systematically pushing severe cases toward one tone band. A corrected estimator returned a median consistent with Fitzpatrick type III and established that the corpus contains no auditable dark-skinned population, restricting all subsequent analysis to two tone bands.",

 "Four findings are reported. First, propagation is real and large: passing every subject through the same deterministic recommender on the true grade and on the predicted grade, a misclassified user receives a list sharing 0.515 of its contents with the correct list against 1.000 when correctly classified, a penalty of 0.485. Second, no accuracy disparity is demonstrated between the two auditable tone bands, though calibration is approximately 30% worse for the smaller group, and the propagation itself is not demonstrably tone-differential. Third, a counterfactual probe shifting rendered skin tone while holding lesion content fixed shows the same model is causally tone-sensitive: 6.8% of predictions flip under a twenty-degree darkening shift, dose-responsively, with a zero-shift control returning exactly zero, and instability is greatest where training tone coverage is sparsest. Fourth, no bias mitigation strategy outperformed the unmitigated baseline, with seed variance under a single strategy exceeding the largest between-strategy difference eightfold. Independently, the review corpus is representationally biased 27.4 to 1 between the lightest and deepest bands, while deeper-toned reviewers raise hyperpigmentation concerns at 2.4 times the rate.",

 "The central result is the conjunction of the second and third findings: a model can be demonstrably tone-sensitive at the mechanism level while showing no disparity in aggregate accuracy or downstream outcome metrics. Disaggregated auditing and causal probing answer different questions, and reporting only the former can license a false assurance of tone-blindness. The work contributes the transfer of pipeline-fairness methodology into a commercially widespread domain where it has been absent, a quantified measurement of bias propagation into recommendation, and a characterisation of the dataset limitations that currently bound fairness auditing of consumer skincare systems.",
]


# --- Chapter 1: Introduction ----------------------------------------------

CH1_INTRO = [
 "Automated skin analysis has moved quickly from research laboratories into consumer products. Major beauty retailers now offer tools that photograph a user's face, infer skin concerns, and recommend products on that basis. These systems reach large audiences, yet the image corpora available to build them under-represent darker skin tones, a deficiency documented repeatedly across publicly available dermatological datasets [wen2022, groh2021].",

 "The consequences of that imbalance are not confined to the classifier. A user whose skin is misassessed does not simply receive a wrong label; they are shown a different set of products, potentially fewer, potentially more expensive, potentially unsuited to their actual needs. The harm is realised at the point of recommendation rather than at the point of classification, and no metric computed on the classifier alone will observe it. This project builds a two-stage system of the kind now deployed commercially, an image-based severity assessment feeding a review-driven recommender, and measures fairness at both stages and across the boundary between them.",
]

CH1_PROBLEM = [
 "Two mature research literatures bear on this problem, and they do not currently meet.",

 "The first audits dermatological image models across skin tone groups, consistently reporting degraded performance on darker skin and tracing it to training corpora dominated by lighter tones [daneshjou2022, kinyanjui2020, adamson2018]. It supplies the methodological grounding for the audit conducted here, but it is uniformly diagnostic: it addresses clinical harm to patients, and it stops at the classifier.",

 "The second examines how bias behaves when models are composed. Fairness established at one stage is not preserved automatically under composition [dwork2019], multi-stage systems can concentrate disadvantage in ways their components do not individually reveal [bower2017], and two-stage architectures constrain downstream fairness through the candidate set the first stage produces [wang2022, ekstrand2022]. Its application domains, however, are predominantly employment, credit and content ranking.",

 "Between them lies the gap this project occupies. Systems recommending skincare from facial images do exist: Lee et al. pair facial condition analysis with cosmetic ingredient analysis and evaluate on recommendation accuracy, treating bias as a limitation for future data collection rather than a quantity to be measured [lee2024]. That work also reports severe imbalance in product concern labelling, which motivates the corpus audit undertaken here.",

 "Pipeline-fairness methodology has therefore not been applied to consumer skincare recommendation, and existing skincare recommenders have not been audited for tone-related disparity at all, let alone for its propagation. The contribution claimed is one of application and evidence rather than of novel method: the transfer of an established apparatus into a domain where it is absent, on a class of system already deployed at commercial scale, and where the affected people are consumers with no clinician positioned to catch the error.",
]

CH1_OBJECTIVES = [
 "The investigation is organised around six research questions, set out in Table 1. RQ2 is the principal question; the remainder either establish the conditions under which it can be answered (RQ1, RQ3), test the same question by a different method (RQ4), or bound the generality of the answer (RQ5, RQ6).",

 ('TABLE', 'The six research questions and the form of evidence each requires.', [
   ["", "Research question", "Form of evidence"],
   ["RQ1", "Does automated acne severity classification perform equitably across skin tone groups?",
    "Disaggregated error rates and calibration, reported as effect sizes with confidence intervals"],
   ["RQ2", "Do classification disparities propagate into downstream recommendation outcomes?",
    "Relevance, list overlap, catalogue coverage and price distribution, conditioned on upstream correctness"],
   ["RQ3", "Which mitigation strategies reduce disparity, and at what cost to predictive performance?",
    "Multi-seed comparison of six strategies against an unmitigated baseline"],
   ["RQ4", "Is the model's output causally sensitive to skin tone itself, independent of condition?",
    "Counterfactual tone shift with lesion content held fixed"],
   ["RQ5", "Is the product review corpus itself representationally biased by reviewer skin tone?",
    "Review volume, rating, price and concern coverage by reported tone, with effect sizes"],
   ["RQ6", "Do findings from a large tone-labelled corpus replicate on a small tone-estimated one?",
    "Cross-arm comparison of disparity direction, mitigation ordering and interval width"],
 ]),

 "Supporting these is a delivery objective: a working demonstrator evidencing the pipeline end to end, architected so that all recommendation logic remains deterministic and inspectable.",
]

CH1_METHODOLOGY = [
 "The methodology is empirical, comparative, and committed in advance. A two-arm design was adopted because no single public dataset supports both statistical power and application realism. Arm B, the applied arm, uses ACNE04, a corpus of 1,457 dermatologist-graded facial photographs annotated under the Hayashi severity criterion [wu2019, hayashi2008]; the model trained on it is the model the demonstrator runs. Arm A was to supply a well-powered reference audit on a corpus carrying ground-truth Fitzpatrick labels [groh2021]. No copy pairing that corpus's images with its tone metadata could be obtained, and Arm A is therefore reported as an unexecuted component with its consequences stated explicitly rather than absorbed silently.",

 "Because facial corpora do not carry tone labels, tone groups are derived computationally using the Individual Typology Angle in CIELAB colour space [chardon1991, delbino2013], and validated by visual inspection of stratified samples. That validation proved consequential rather than procedural, as Chapter 3 records.",

 "A ResNet-50 baseline [he2016] is compared against EfficientNet and Vision Transformer architectures [tan2019, dosovitskiy2021] under an identical protocol across multiple random seeds, so that claimed differences can be separated from run-to-run variance, a distinction the benchmarking literature shows is routinely underestimated [bouthillier2021]. Disparities are reported as effect sizes with bootstrapped confidence intervals [efron1993, cohen1988], and calibration alongside accuracy [guo2017].",

 "Propagation is measured by passing every subject through the same deterministic recommender twice, once on the dermatologist's grade and once on the model's prediction, so that any divergence between the two lists is attributable to the classifier alone. A counterfactual probe complements this by shifting rendered skin tone while holding lesion content fixed, supplying causal evidence correlational auditing cannot [kusner2017]. The metric set and group definitions were fixed before results were inspected, so that reporting could not drift toward whichever measure proved most flattering.",
]

CH1_LEGAL = [
 "The system is a cosmetic recommendation tool and not a medical device. It does not diagnose, and this is stated in the interface as well as throughout this report. Two behaviours enforce that boundary in software rather than in documentation alone: the classifier abstains below a confidence threshold instead of producing a low-confidence guess, with the threshold derived from the calibration analysis rather than chosen arbitrarily [geifman2017]; and the interface declines medical requests, including diagnosis, malignancy assessment and prescription guidance, redirecting the user to qualified care. That refusal behaviour was specified in advance and tested against a purpose-built adversarial set, with transcripts in the appendices.",

 "All data are pre-existing public research datasets [wu2019, sephora2023]. No new facial images were collected and no participants recruited, so no primary-data ethical approval was required, though departmental guidance was followed. Facial images are personal data under the General Data Protection Regulation and are handled accordingly: secure local storage, no redistribution, and no retention of images submitted to the demonstrator, which processes an upload and discards it.",

 "Skin tone is treated as a sensitive attribute used solely for fairness measurement, and is never an input to the recommendation. Measuring disparity requires the attribute, whereas acting on it would constitute the harm the measurement exists to detect.",

 "A professional consideration runs alongside these. An audit reporting only aggregate parity can license a false assurance, and the central result of this project is precisely such a case. Reporting it honestly, including three null results a less careful framing could have concealed, is treated as an obligation of the work rather than a weakness of it [selbst2019, mitchell2019].",
]

CH1_BACKGROUND = [
 "Three technical ideas underpin what follows. The first is ordinal severity grading: acne is assessed clinically on ordered categories, and the Hayashi criterion annotating ACNE04 defines four grades from mild to very severe [hayashi2008]. Because the scale is ordinal, adjacent-grade confusion is the dominant error mode, and the treatment classes endorsed by dermatological guidance at adjacent grades overlap substantially [zaenglein2016]. Both facts shape how recommendation error must be measured.",

 "The second is skin tone measurement. The Fitzpatrick scale remains the dermatological reference standard, though it was designed to describe sun sensitivity rather than pigmentation and is a coarse instrument for fairness work [fitzpatrick1988, groh2022]. Where labels are unavailable, the Individual Typology Angle offers an objective alternative computed from lightness and yellow-blue chromaticity in CIELAB space [chardon1991]. It is a proxy, sensitive to illumination and to anything altering skin colour locally, and is treated as such throughout.",

 "The third is calibration. A classifier's confidence is meaningful only if it corresponds to its accuracy, and modern networks are frequently over-confident [guo2017]. Calibration is reported per group because a model may be equally accurate on two groups while being more confidently wrong on one, a difference invisible to accuracy metrics but directly consequential for any system using confidence to decide whether to advise at all.",
]

CH1_STRUCTURE = [
 "Chapter 2 reviews the three literatures the project sits between and surveys the technologies used. Chapter 3 describes the implementation: corpus preparation, the derivation and correction of the tone estimator, the severity classifier, the severity-to-ingredient mapping, the deterministic recommender, and the demonstrator. Chapter 4 presents the evaluation, addressing each research question in turn and positioning the findings against related work. Chapter 5 concludes, sets out future work, and reflects on the conduct of the project.",
]


# --- Chapter 2: Literature - Technology Review ----------------------------

CH2_INTRO = [
 "This chapter reviews the work the project builds on and positions its contribution against it. Three literatures are relevant and are treated in turn: fairness auditing of dermatological image models, fairness across composed machine learning pipelines, and automated skincare recommendation. The gap the project occupies lies at their intersection. A technology review follows, covering the frameworks and libraries used and the reasoning behind each selection.",
]

CH2_LIT = [
 "2.1 Fairness in dermatological and facial image models",

 "Deep learning reached dermatologist-level performance on skin lesion classification with Esteva et al., who matched board-certified clinicians on biopsy-proven cases [esteva2017]. Attention turned quickly to whether that performance was evenly distributed. Buolamwini and Gebru had already established the pattern in a neighbouring domain, showing commercial gender classifiers erred far more often on darker-skinned women than on lighter-skinned men, and demonstrating that aggregate accuracy can conceal severe subgroup failure [buolamwini2018]. Adamson and Smith raised the same concern for dermatology directly, arguing that models trained on predominantly light-skinned corpora would extend existing health disparities rather than reduce them [adamson2018].",

 "Subsequent work supplied the measurements. Groh et al. released Fitzpatrick17k, annotating roughly 17,000 clinical images with Fitzpatrick tone labels, and found models trained on it performed best on the tone groups best represented within it [groh2021]. Daneshjou et al. assembled a diverse, curated clinical set and reported that published algorithms lost accuracy on darker skin even when they performed well overall [daneshjou2022]. Kinyanjui et al. characterised the imbalance in the widely used ISIC corpora and analysed classifier behaviour across tone [kinyanjui2020]. Wen et al. reviewed publicly available skin cancer image datasets systematically and found tone metadata largely absent: most datasets do not record skin tone at all, so the disparity frequently cannot be measured even in principle [wen2022]. This is the practical consequence of the documentation gap Gebru et al. address with datasheets for datasets, which argue that provenance and composition should be recorded at publication precisely because downstream users cannot reconstruct them later [gebru2021]. Mitigation approaches have followed, including tone detection and debiasing applied to lesion classification [bevan2022].",

 "Two methodological problems recur and bear directly on this project. The first is that the Fitzpatrick scale, designed to describe sun sensitivity rather than pigmentation, is a coarse and partly subjective instrument [fitzpatrick1988]; Groh et al. later quantified this by comparing expert, crowd and algorithmic tone annotations of the same images and finding substantial disagreement [groh2022]. The second is that where labels are absent they must be estimated from pixels, an approach used at scale in face analysis [merler2019] and grounded dermatologically in the Individual Typology Angle [chardon1991, delbino2013]. Estimation introduces its own failure modes, and Chapter 3 documents one that materially altered this project's direction.",

 "This literature supplies the methodological grounding for RQ1, RQ3 and RQ4. Its orientation, however, is uniformly diagnostic: the harm considered is clinical misdiagnosis, the subject is a patient, and the analysis ends at the classifier's output. What a model's error does after it leaves the model is not its concern.",

 "2.2 Fairness across composed pipelines",

 "A separate literature asks what happens when models are combined. Its formal foundations lie in the group and individual fairness definitions of Dwork et al. [dwork2012] and Hardt et al. [hardt2016], and its central negative result is that these properties do not survive composition: Dwork and Ilvento showed that components each satisfying a fairness criterion can produce a system that satisfies none, and that fairness must therefore be reasoned about at the level of the composed system [dwork2019]. Bower et al. reached a compatible conclusion for sequential decision pipelines, where disadvantage accumulates across stages in ways no single stage exhibits [bower2017].",

 "Recommender systems are the application area where this matters most concretely, because they are almost always multi-stage. Wang and Joachims analysed two-stage architectures in which a fast, less accurate retrieval stage determines the candidate set a more careful ranker can access, showing that fairness at the second stage is bounded by what the first stage surfaces [wang2022]. Ekstrand et al. survey the broader field of fairness in information access, distinguishing harms to consumers from harms to providers [ekstrand2022]. Related work on popularity bias documents how rankers concentrate exposure on a small proportion of the catalogue [abdollahpouri2019], and coverage and diversity have long been argued as necessary complements to accuracy in recommender evaluation [ge2010]. Both concerns recur in the results reported in Chapter 4.",

 "Running alongside is work on how fairness should be measured at all. Mehrabi et al. survey the proliferation of definitions [mehrabi2021], Corbett-Davies and Goel argue several widely used criteria are poorly motivated and can harm the groups they intend to protect [corbett2018], and Barocas et al. provide the standard synthesis [barocas2023]. Selbst et al. caution that fairness is not a property of a model considered in isolation but of the sociotechnical system it sits within, and that abstracting the model away from its deployment context is itself a failure mode [selbst2019]. That argument is the direct justification for measuring this pipeline end to end rather than auditing its classifier alone.",

 "Mitigation methods relevant to RQ3 come from the same tradition. Group distributionally robust optimisation targets worst-group rather than average performance [sagawa2020], and focal loss reweights training toward hard examples [lin2017]. Both are compared here against reweighting and resampling. A caution attaches to any such comparison: Bouthillier et al. demonstrate that benchmark differences are routinely reported without accounting for the variance introduced by random seeds and data ordering [bouthillier2021], and Picard shows that seed choice alone can move published-looking results on standard vision benchmarks [picard2021]. This concern proved decisive in the mitigation experiment reported in Chapter 4.",

 "The application domains of this literature, however, are predominantly employment, credit, and content ranking. It has not been applied to consumer skincare.",

 "2.3 Automated skincare recommendation",

 "Systems that recommend cosmetic products from facial images have been built and published. Lee et al. present the closest comparator to the pipeline constructed here, combining facial skin condition analysis with an ingredient-analysis model to produce personalised product recommendations [lee2024]. The system is evaluated on recommendation quality. Bias is acknowledged as a limitation to be addressed through broader future data collection rather than treated as a quantity to be measured, and no disaggregated evaluation is reported. The same paper documents a severe imbalance in product-level concern labelling, with acne-labelled products numbering in the low hundreds against several thousand labelled for ageing concerns, a finding this project independently reproduces on a different catalogue in Chapter 4.",

 "Recommendation here rests on a clinical rather than a behavioural signal, which shapes the design. Dermatological guidance defines which treatment classes are appropriate at which severity, with topical agents such as salicylic acid, benzoyl peroxide, retinoids and azelaic acid indicated according to grade [zaenglein2016, arif2015], and niacinamide supported for barrier and pigmentation concerns [bissett2005]. Severity itself is graded on ordered clinical criteria [hayashi2008]. Because these relationships are documented, the mapping from assessed severity to endorsed ingredient classes can be made explicit and inspectable rather than learned, which is what allows propagation to be attributed to the classifier rather than to the ranker.",

 "One concern is specifically relevant to tone. Post-inflammatory hyperpigmentation, the dark marks left after acne resolves, occurs more frequently and persists longer in skin of colour, and is often the presenting complaint rather than the acne itself [davis2010]. A recommender that reasons only about lesion severity will under-serve it, and a review corpus that under-represents darker-skinned reviewers will carry thin evidence about it. RQ5 measures exactly this.",

 "2.4 The gap",

 "The three literatures do not meet. Dermatological fairness work stops at the classifier and addresses patients; pipeline fairness work reasons about composition but in unrelated domains; skincare recommenders are built and evaluated on accuracy without disaggregation. Pipeline-fairness methodology has therefore not been applied to consumer skincare recommendation, and published skincare recommenders have not been audited for tone-related disparity, let alone for its propagation into what users are shown. The contribution claimed here is the transfer of an established methodological apparatus into that gap, together with the evidence it produces. It is a contribution of application rather than of method, and is claimed as such. A secondary methodological contribution follows from the results: the demonstration that disaggregated auditing and causal probing can disagree on the same model, and that reporting only the former can license a false assurance.",
]

CH2_TECH = [
 "The implementation uses established open-source components throughout. Selection favoured tools whose behaviour is inspectable, since the project's purpose is measurement rather than performance.",

 "Modelling is built on PyTorch [paszke2019], with pretrained image architectures obtained through the timm library, which supplies a consistent interface across model families and standardised ImageNet weights [wightman2019]. This matters for the architecture comparison: ResNet-50 [he2016], EfficientNet [tan2019] and Vision Transformer [dosovitskiy2021] are trained through identical code paths, so differences are attributable to the models rather than to divergent implementations. All are initialised from ImageNet pretraining [deng2009], without which a 1,457-image corpus would be far too small to train a modern vision model.",

 "Text processing uses the Hugging Face transformers library [wolf2020] with DistilBERT [sanh2019], a compressed variant of BERT [devlin2019] retaining most of its accuracy at roughly half the parameters. The reduction was decisive under the compute available: a full transformer [vaswani2017] fine-tune over 120,000 reviews was not affordable, whereas the distilled model was. Classical metrics, resampling utilities and the statistical machinery for the audits come from scikit-learn [pedregosa2011], with NumPy [harris2020] and pandas [mckinney2010] underlying the data layer. Image handling and the CIELAB colour conversions on which tone estimation depends use OpenCV [bradski2000].",

 "The demonstrator is a Streamlit application [streamlit2019]. The choice was governed by the architectural constraint rather than by features: the interface had to be assembled from ordinary Python so that the recommendation path could be held entirely outside any generative component, and Streamlit permits this while requiring no front-end work that would displace research time.",

 "Model training was performed on Kaggle's hosted GPU environment, the local machine having no CUDA device. This shaped the workflow more than the results. Notebooks are held as version-controlled Python sources and generated into notebook form for upload, so that experiment code remains reviewable and diffable rather than living only inside notebook JSON. Every training notebook carries a smoke-test flag permitting a short validation run before a full commit, a precaution adopted after an interactive session's outputs were lost.",
]

CH2_SUMMARY = [
 "Three literatures bear on the problem and none addresses it. Dermatological fairness auditing supplies rigorous methods for measuring disparity in image models but applies them to diagnosis and stops at the classifier. Pipeline fairness supplies the theory of how bias behaves under composition but is situated in employment, credit and content ranking. Automated skincare recommendation builds the system class in question but evaluates it on accuracy alone. The project takes the methods of the first, the analytical frame of the second, and the application of the third, and reports what results. The technologies selected serve that purpose: pretrained vision models to make the classifier credible, a distilled transformer to make text modelling affordable, and a deterministic ranker in ordinary Python to keep the seam between the two stages open to inspection.",
]

# --- Chapter 3: Implementation --------------------------------------------

CH3_IMPL = [
 "This chapter describes how the system was built: the corpora and their preparation, the derivation and correction of the skin tone estimator, the severity classifier, the mapping table connecting assessment to ingredients, the recommender, the counterfactual probe, and the demonstrator. Design decisions are given with the reasoning behind them; measurements appear only where a decision depended on one.",

 "3.1 Corpora and preparation",

 "Two corpora are integrated into the pipeline. ACNE04 supplies 1,457 facial photographs graded by dermatologists under the Hayashi criterion into four ordered severity classes [wu2019, hayashi2008]. The severity distribution is imbalanced: 497 images at grade 0, 637 at grade 1, 186 at grade 2 and 137 at grade 3, so the two severe grades together account for 22.2% of the corpus. That imbalance is not corrected away, because it is a property of the problem, but it is carried forward explicitly through class weighting and macro-averaged reporting.",

 "The Sephora corpus supplies 8,494 products, of which 2,420 are skincare, and 1,094,411 customer reviews with ingredient lists, prices, star ratings, review text and reviewer-reported attributes including skin type and skin tone [sephora2023]. Reviewer-reported tone is present for 923,802 reviews, or 84.4%.",

 "[[FIG:price_distribution.png|Skincare product price in USD, capped at 300 for display. The distribution peaks near 35 USD with a long right tail, confirming the catalogue is prestige-only: there is no mass-market segment below roughly 10 USD.]]",

 "The price distribution bounds what the recommender's budget filter can do. The catalogue is prestige-only and contains no mass-market products, which is stated as a boundary condition rather than corrected: the recommender's scope is premium skincare, and this constrains the price analysis rather than invalidating it.",

 "[[FIG:rating_vs_volume.png|Product mean rating against review count on a logarithmic scale. Ratings compress toward 4.0-4.7 as volume rises, so the rating signal discriminates poorly among well-reviewed products.]]",

 "Two properties of the review signal shape the ranking function. Ratings compress sharply as review volume grows, with almost every well-reviewed product falling between 4.0 and 4.7, so star rating alone separates products weakly; it is given a low weight in Section 3.5 for this reason. Products with fewer than five reviews are excluded entirely, since the wide rating spread at low volume reflects sampling noise rather than quality.",

 "[[FIG:skin_type_mix.png|Review counts by reviewer-reported skin type. Combination skin accounts for roughly half of all reviews, and oily skin the fewest.]]",

 "[[FIG:skin_tone_representation.png|Review counts across the fourteen self-reported skin tone descriptors in the raw corpus. The vocabulary is granular and unevenly populated, which is why tone is collapsed into three ordered bands before analysis.]]",

 "Reviewer attributes are self-reported and unevenly distributed. Skin type is dominated by combination skin, which matters because the recommender's skin-type term is derived from these labels. Skin tone is recorded against fourteen descriptors of varying granularity, several of them near-empty; these are collapsed into the same three ordered bands used for the image analysis, so that the two halves of the project speak a common vocabulary and so that per-band counts support interval estimates. That collapse is a judgement, and the raw distribution is reported alongside it so the reader can see what was merged.",

 "The three corpora share no join key, and none is contrived. The image arm connects to the product arm solely through the severity-to-ingredient mapping described in Section 3.4. Keeping that seam explicit is what later permits divergence in recommendations to be attributed to the classifier rather than to an opaque association learned across datasets.",

 "3.2 Deriving skin tone, and correcting the estimator",

 "Neither facial corpus supplies tone labels, so tone groups are derived computationally using the Individual Typology Angle in CIELAB colour space, following established dermatological practice [chardon1991, delbino2013]. The angle is computed from lightness and yellow-blue chromaticity over skin-masked pixels, and the resulting values are binned into ordered bands.",

 "The first implementation used a permissive skin mask with percentile trimming on luminance. It returned a distribution in which 59.5% of ACNE04 fell into the darkest band. ACNE04's population is predominantly Chinese, so this result is not credible on its face, and it was treated as a measurement failure rather than as a finding.",

 "Validation followed the procedure committed to in the proposal: manual inspection of a stratified sample. Montages of randomly sampled faces grouped by estimated tone make the failure immediately visible, and two distinct error modes are apparent.",

 "[[FIG:ita_montage_method_A.png|Faces sampled from each tone band under the initial estimator. The band labelled darkest contains ordinary light-to-medium skinned faces, several evenly lit; the lightest band is dominated by tightly cropped, heavily inflamed images. Both patterns identify measurement error rather than corpus composition.]]",

 "First, the darkest band contains ordinary subjects. Faces assigned to it are unremarkable light-to-medium skinned faces by any reasonable visual judgement. The estimator was not mis-sorting a genuine minority; it was under-reading the angle across the whole corpus, because averaging over all skin-masked pixels incorporates shadowed peripheral regions, hair boundaries and dim ambient lighting, all of which depress median luminance.",

 "Second, inflamed close-ups are pushed the other way. The lightest band is dominated by tightly cropped, heavily inflamed images. Erythema raises the red-green and yellow-blue channels, and because the angle is computed as the arctangent of lightness over yellow-blue chromaticity, inflammation inflates the estimate. Severity therefore contaminated the tone measurement in precisely the direction that would most damage a fairness audit: the most severe cases were being systematically assigned to one tone band, which would have manufactured an apparent tone disparity out of a severity artefact.",

 "Two refinements were tested against the same visual standard. Method B discards the reddest quartile of skin pixels, suppressing the contribution of inflamed lesions, and retains the brighter half of the remainder, suppressing shadow. Method C applies a central crop before the same procedure, restricting measurement to the cheek and forehead region.",

 ('TABLE', 'The three tone estimators compared. Method B was adopted; Method C is retained as a sensitivity variant.', [
   ["Method", "Pixel selection", "Median ITA", "Light / Medium / Deep"],
   ["A (initial)", "Skin mask, luminance trimmed", "14.8 deg", "4% / 36% / 59%"],
   ["B (adopted)", "Drop reddest 25%, keep brighter half", "37.3 deg", "37% / 50% / 13%"],
   ["C (variant)", "Central crop, then Method B", "45.8 deg", "67% / 26% / 7%"],
 ]),

 "Method B returns a median angle of 37.3 degrees, corresponding to Fitzpatrick type III, which is consistent with the corpus demographic. It was adopted for all subsequent analysis.",

 "[[FIG:ita_method_distributions.png|Distribution of estimated Individual Typology Angle under each of the three methods. All three retain a left tail below minus fifty degrees, which no pixel-selection rule removes.]]",

 "The distributions supply a further check. All three methods retain a left tail extending below minus fifty degrees. Values that extreme are not plausible skin measurements for any subject in this corpus, and their persistence across estimators indicates a floor of irreducible measurement noise arising from illumination rather than from pigmentation. Method B compresses the bulk of the distribution into a plausible range while leaving that tail largely intact, consistent with the tail being an artefact of image capture. Agreement between methods B and C on band assignment is 60.1%, which is itself a caution against treating any single implementation of this measure as authoritative.",

 "[[FIG:ita_montage_method_B.png|The same sampling procedure under the adopted estimator. Band assignments are visually coherent, and the confound between severity and estimated tone is no longer apparent.]]",

 "[[FIG:ita_montage_method_C.png|The sensitivity variant, which restricts measurement to a central crop before applying the same pixel selection. It shifts the distribution lighter still, and agrees with the adopted estimator on band assignment for only 60.1% of images.]]",

 "The correction changed the shape of the project. Under the corrected estimator the corpus contains no population of genuinely dark-skinned subjects, which triggered the pre-committed decision gate described in Section 4.1 and restricted all subsequent tone analysis to two bands.",

 "3.3 The severity classifier",

 "A ResNet-50 pretrained on ImageNet was fine-tuned for four-class severity classification [he2016, deng2009]. Given the corpus size, evaluation uses repeated stratified five-fold cross-validation rather than a single held-out split, so that every image contributes an out-of-fold prediction and disparity estimates draw on the full corpus rather than on a 292-image test partition. Splits are stratified by severity class and by tone band simultaneously, so that per-group evaluation remains possible in every fold.",

 "Class weighting is applied during training to counter the severity imbalance. Augmentation is deliberately restricted: geometric transformations are applied freely, but colour and brightness transformations are not, because they would distort the very signal the fairness analysis depends on. Augmenting tone would have made the tone measurement meaningless.",

 "The out-of-fold predictions produced here are reused throughout. They supply the disaggregated audit in Section 4.3, the propagation analysis in Section 4.4, and the abstention threshold in Section 3.9. Using one consistent set of predictions across all three keeps the analyses commensurable, and means the deployed threshold is derived from the same evidence as the audit that motivates it.",

 "The architecture comparison in Section 4.2 trains four architectures across three seeds under an identical protocol. A single learning rate is used throughout, selected for the ResNet-50 baseline and held constant so that differences would be attributable to architecture rather than to tuning. That decision has a cost, discussed with the result.",

 "3.4 Ingredient parsing and the mapping table",

 "The seam between the two stages is a documented mapping from assessed severity and declared attributes to ingredient classes. It is a design artefact, and it is presented as one.",

 "Ingredient parsing is rule-based rather than learned. The ingredient field is inconsistent free text carrying marketing preamble, nested list encodings and variable ordering; a deterministic parser makes every match inspectable, whereas a learned extractor would obscure exactly the seam the fairness analysis needs to examine. Patterns are deliberately conservative, because a false negative costs a missed candidate while a false positive puts an unsuitable product in front of a user. Parsing normalises free text into a controlled vocabulary of 28 ingredient classes and succeeds on 2,281 of 2,420 skincare products, or 94.3%.",

 "One corpus-level observation from parsing bounds what the recommender can achieve, and is reported as a finding in its own right: only 473 products, 19.5% of the skincare catalogue, carry any acne-directed active ingredient, against 1,339 products, 55.3%, carrying an anti-ageing or hydration active. This independently reproduces on a different catalogue the concern-labelling imbalance Lee et al. report [lee2024], and it constrains differentiation most at the severe grades, where the endorsed ingredient classes are narrowest.",

 "The mapping itself follows dermatological guidance rather than marketing sources [zaenglein2016, arif2015, bissett2005]. Weights express preference and are used to score candidates; they do not prescribe, and no prescription-strength agent is ever recommended. The retinoid class refers throughout to cosmetic retinol-class ingredients available in a retail catalogue.",

 ('TABLE', 'The severity-to-ingredient mapping. Grades 2 and 3 additionally attach a referral flag.', [
   ["Grade", "Severity", "Preferred ingredient classes", "De-prioritised", "Referral"],
   ["0", "Mild", "Salicylic acid, niacinamide, gentle AHA, PHA", "High-strength benzoyl peroxide", "No"],
   ["1", "Moderate", "Salicylic acid, benzoyl peroxide, retinoid, azelaic acid, niacinamide", "Heavy occlusives", "No"],
   ["2", "Severe", "Benzoyl peroxide, retinoid, azelaic acid, niacinamide", "Pure hydrators alone", "Yes"],
   ["3", "Very severe", "Benzoyl peroxide, retinoid, azelaic acid", "Cosmetic-only actives", "Yes"],
 ]),

 "Attributes the image cannot supply are elicited conversationally and never inferred from pixels. Declared dryness pulls toward ceramides, hyaluronic acid, glycerin, squalane and panthenol and away from denatured alcohol and high-strength exfoliants. Declared sensitivity pulls toward centella, colloidal oatmeal, allantoin and panthenol and away from fragrance and essential oils. Declared oiliness pulls toward clay, zinc, niacinamide and salicylic acid. A declared concern with dark marks pulls toward niacinamide, azelaic acid, vitamin C, tranexamic acid and alpha arbutin, and away from irritants that worsen inflammation.",

 "That last row is the one this project is most concerned with. Post-inflammatory hyperpigmentation disproportionately affects darker skin [davis2010], and the corpus audit in Section 4.7 finds it raised at more than twice the rate by the reviewers least represented in the corpus. Inferring it from an image was rejected: no public facial dataset labels it, and estimating a pigmentation concern from pixels in a system whose tone sensitivity is under investigation would be indefensible. It is therefore declared, not detected.",

 "Adjacent grades share endorsed classes, which is clinically correct and has a measurement consequence recorded in Section 4.4: relevance measured against a permissive reference standard saturates, and list overlap becomes the informative metric.",

 "3.5 The recommender",

 "The ranker is content-based and wholly deterministic. A learned collaborative-filtering approach was rejected at the design stage on three grounds: the data do not support it, lacking a dense user-item interaction matrix; it would consume disproportionate time; and an opaque ranker would obstruct the fairness analysis that is the project's purpose. Determinism is not a simplification here but a precondition, because it is what makes divergence between two recommendation lists attributable to the classifier alone.",

 "The candidate set is filtered to products with a parsed ingredient list, in a skincare category, and carrying at least five reviews, so that the review-derived signals are not computed from a handful of ratings. Every product is then scored as a weighted sum of five named components.",

 ('TABLE', 'Components of the ranking score. Weights are documented and fixed, chosen for interpretability rather than tuned for accuracy.', [
   ["Component", "Weight", "Source"],
   ["Ingredient match", "1.00", "Mapping-table weights against parsed ingredient flags, normalised"],
   ["Review sentiment", "0.35", "Share of positive reviews for the product"],
   ["Star rating", "0.25", "Mean rating, normalised to the unit interval"],
   ["Skin-type fit", "0.20", "Reviewer-reported suitability for the declared skin type"],
   ["Irritancy penalty", "-0.30", "Applied only when sensitivity is declared"],
 ]),

 "Weights were fixed by design rather than tuned against an outcome metric. Tuning them would have optimised the ranker toward whichever relevance measure was used to evaluate it, making the evaluation circular. Every score decomposes into these named terms, so any recommendation can be explained by inspection, and a budget constraint is applied as a filter rather than as a score term so that price never trades against clinical suitability.",

 "A popularity-only ranker, ordering by a normalised log of the catalogue's favourite counts, is retained as a baseline. Its purpose is to establish that the mapping table contributes something beyond surfacing best-sellers, a comparison recommended in the recommender evaluation literature [ge2010, abdollahpouri2019].",

 "Relevance is measured as precision at k against the set of ingredient classes the mapping table endorses for a subject's true grade. There is no ground truth for the correct product, so evaluation uses a defensible reference standard of acceptable ingredient classes rather than exact product matches.",

 "3.6 Measuring propagation: the two-pass design",

 "The propagation measurement reported in Section 4.4 rests on a design decision made here rather than at analysis time, and it is the reason the ranker had to be deterministic.",

 "Each of the 1,457 subjects is passed through the same recommender twice. The first pass uses the dermatologist's grade and yields the recommendation the subject should have received. The second uses the model's out-of-fold prediction and yields the recommendation the deployed system would actually produce. The two lists are then compared directly, by list overlap, by relevance against the reference standard, by price, and by whether a referral flag was raised in one pass but not the other.",

 "The design's value lies in what it excludes. Because the mapping table and every term in the ranking function are deterministic, and because the declared attributes are held identical across both passes for a given subject, the only quantity that differs between them is the severity grade. Any divergence between the two lists is therefore attributable to the classifier alone, with no residual to apportion between upstream error and ranker variance. Had the ranker contained any learned or stochastic component, the two passes would have differed for reasons the analysis could not separate, and the measurement would have been uninterpretable. This is the concrete reason a learned ranker was rejected in Section 3.5, and it is what makes the propagation figure a measurement rather than an estimate.",

 "Attributes the image cannot supply, namely skin type, sensitivity, dark-mark concern and budget, are sampled for each simulated subject from the empirical distribution of self-reported reviewer attributes in the review corpus, under a fixed seed. Sampling from the empirical distribution rather than assigning a uniform default gives each subject a distinct and realistic profile, so the recommender is exercised across the range of attribute combinations it would meet in deployment rather than at a single point. The fixed seed makes the whole analysis reproducible. That these profiles are simulated rather than collected from real users is a limitation, recorded as such: the propagation measurement is valid for the population of attribute combinations the review corpus represents.",

 "A popularity-only baseline is run through the identical harness, so that the comparison in Section 4.4 isolates the contribution of the mapping table rather than of the evaluation procedure.",

 "3.7 The counterfactual tone probe",

 "The disaggregated audit is correlational: it compares groups differing in many respects at once. The probe intervenes directly, following the counterfactual formulation of fairness [kusner2017].",

 "Each image is rendered at five tone levels. Only the lightness channel is shifted, and only on skin-masked pixels, by the offset required to move that face's angle to a target of minus twenty, minus ten, zero, plus ten and plus twenty degrees. The yellow-blue channel is held fixed while solving for lightness. The red-green channel, which carries acne erythema, is never modified, and spatial structure is untouched, so lesion count, shape and position are identical across the ladder. Any change in prediction is therefore attributable to the rendered tone alone.",

 "Two controls are built in. The zero-shift arm passes an unmodified image through the identical rendering path, so any nonzero result there would indicate the pipeline itself introduced variation. And because the ladder is graded rather than binary, a dose-response relationship can be tested, which is materially stronger evidence than a single significant contrast.",

 "The design has a boundary that Section 4.6 states plainly: it manipulates a rendered proxy for tone, not the biological and photographic covariates accompanying tone in real subjects. It establishes sensitivity to the tone signal as encoded in pixels; it does not reproduce a genuinely different person.",

 "3.8 The review corpus pipeline",

 "The review corpus is processed independently of the image pipeline, and that independence is deliberate. Had the corpus audit depended on the image model, the compressed tone range documented in Section 3.2 would have bounded it too, and the project would have carried a single point of failure. Keeping the two arms separate means a substantive fairness contribution survives regardless of what the image corpus turns out to support.",

 "Reviewer-reported skin tone is normalised from the corpus's free-form tone descriptors into the same three ordered bands used for the image analysis, so that findings from the two arms can be discussed in a common vocabulary. Reviews without a usable tone descriptor are excluded from tone-disaggregated analysis rather than imputed, and the excluded proportion is reported.",

 "Concern detection uses the same rule-based approach as ingredient parsing, and for the same reasons. Mentions of post-inflammatory hyperpigmentation are identified through a documented pattern set covering dark marks, scarring and discolouration vocabulary. A learned classifier would have been more flexible, but the quantity being measured is a rate difference between groups, and an opaque detector with its own uneven error profile across those groups would confound exactly the comparison it was built to support.",

 "A DistilBERT sentiment classifier [sanh2019] is fine-tuned on 120,000 label-balanced reviews. Labels are derived from star ratings, with three-star reviews dropped as ambiguous, and the training set is balanced so that the classifier cannot reach high accuracy by exploiting the corpus's positive skew. The purpose is diagnostic rather than functional: the recommender uses a rating-derived sentiment signal, and the transformer exists to test whether that proxy is sound and whether the text route carries a disparity of its own. Both questions are answered in Section 4.8. This is the only component in the project trained on text, and it sits outside the recommendation path.",

 "3.9 The demonstrator, abstention, and architectural separation",

 "The demonstrator is a Streamlit application [streamlit2019] presenting image upload, severity assessment, conversational attribute elicitation, and explained recommendations.",

 "The classifier abstains below a confidence threshold rather than emitting a low-confidence guess, following the selective classification framework [elyaniv2010, geifman2017]. The threshold was derived rather than chosen: using the same out-of-fold predictions as the audit, it is set to the lowest value achieving 85% selective accuracy while retaining at least 50% coverage.",

 "[[FIG:abstention_curve.png|Selective accuracy against coverage as the confidence threshold varies. The derived operating point trades roughly a third of coverage for an accuracy gain of 8.5 percentage points.]]",

 "Because abstention is itself an allocation, coverage was checked per tone group before the threshold was adopted. Had abstention fallen materially harder on one group, the system would have been quietly denying service to it, an allocative harm invisible to accuracy metrics. The measured gap is reported in Section 4.10.",

 "The final build runs the rules-based dialogue manager with no language model in the loop. This is the strongest available form of the separation the design requires. The proposal specified that a language model could phrase questions, parse free-text replies and verbalise output, but never select, reorder, filter or introduce products; removing it entirely makes that constraint structural rather than behavioural. The interface cannot introduce a product, a claim or a diagnosis, because no generative component exists anywhere in the recommendation path. Every recommendation traces deterministically from classifier, through the documented mapping table, to the ranking function.",

 "That property is not merely a safety feature. It is what makes the propagation analysis in Section 4.4 a measurement of the system it claims to measure: had a generative component sat between the classifier and the user, divergence between recommendation lists could not have been attributed to the classifier.",

 "Refusal behaviour is specified in advance rather than left to a model's discretion. Medical requests, including diagnosis, malignancy assessment and prescription guidance, are declined and redirected to qualified care, and grades 2 and 3 attach a referral flag. This behaviour was tested against a purpose-built adversarial set of 21 medical requests spanning direct, indirect, role-play, smuggled, procedural and emergency framings, alongside 12 benign cosmetic controls included because over-refusal is a genuine failure mode and a system that declines everything is safe and useless. The first evaluation failed, and Section 4.10 reports what it found and what was changed.",
]

# --- Chapter 4: Evaluation and Results ------------------------------------

CH4_EVAL = [
 "This chapter reports the evaluation. Each research question is addressed in turn, with effect sizes and bootstrapped confidence intervals throughout [efron1993]. Three of the questions return null results. These are reported as such, in the form the pre-committed protocol requires, with interval widths given so that the power of each test is visible rather than asserted.",

 "4.1 The skin tone gate",

 "The proposal committed to an explicit decision gate before modelling, the corpus's tone range being load-bearing for RQ1 and RQ2. Under the corrected estimator the outcome is unambiguous, and is reported as a substantive result rather than an obstacle: ACNE04 contains no population of genuinely dark-skinned subjects. Under Method B the darkest band holds 13% of images, but montage inspection confirms this residual is the same lighting and inflammation artefact identified in Section 3.2 rather than Fitzpatrick V or VI skin, and the finding holds under all three estimators.",

 "The gate outcome is therefore no-go for the primary path, with three consequences. Tone-disaggregated analysis is restricted to the Light and Medium bands. The applied arm becomes an under-powered replication rather than a primary audit. And the compressed range is itself reported as a quantified finding: a dataset widely used for automated acne grading cannot support a fairness audit across the tone range such systems meet in deployment. That generalises the concern Wen et al. raise about missing tone metadata [wen2022] to a corpus where the metadata could be estimated but the underlying diversity is absent.",

 "4.2 The severity classifier and architecture comparison",

 "The ResNet-50 baseline achieves out-of-fold accuracy 0.766 and macro-F1 0.746, stable across folds at 0.723 to 0.774. Performance tracks class frequency, with grade 2 weakest; class weighting appears to have protected grade 3, the rarest class, at the expense of its ordinal neighbour.",

 ('TABLE', 'Per-class out-of-fold performance of the severity classifier.', [
   ["Class", "Precision", "Recall", "F1", "Support"],
   ["0 (mild)", "0.81", "0.83", "0.82", "497"],
   ["1 (moderate)", "0.79", "0.75", "0.77", "637"],
   ["2 (severe)", "0.58", "0.66", "0.62", "186"],
   ["3 (very severe)", "0.79", "0.77", "0.78", "137"],
 ]),

 ('TABLE', 'Architecture comparison. Four architectures, three seeds each, identical protocol.', [
   ["Architecture", "Params (M)", "Accuracy", "Macro-F1", "Severe F1", "ECE"],
   ["ResNet-50", "23.5", "0.753 +/- 0.037", "0.731 +/- 0.026", "0.582", "0.108"],
   ["EfficientNet-B0", "4.0", "0.715 +/- 0.040", "0.690 +/- 0.039", "0.549", "0.190"],
   ["EfficientNet-B3", "10.7", "0.709 +/- 0.017", "0.679 +/- 0.030", "0.509", "0.183"],
   ["ViT-Base", "85.8", "0.636 +/- 0.047", "0.639 +/- 0.038", "0.492", "0.085"],
 ]),

 "[[FIG:arch_comparison.png|Macro-F1 by architecture across three seeds. The spread between architectures is roughly three times the pooled seed standard deviation, so the comparison resolves.]]",

 "The spread across architectures, 0.092, is approximately three times the pooled seed standard deviation of 0.033, so the comparison resolves, and in favour of the baseline. Capacity does not help: the largest model is the worst by a wide margin, because Vision Transformers lack the convolutional inductive biases of locality and translation equivariance and compensate with data [dosovitskiy2021], which at 1,093 training images is unavailable. On a corpus this size the binding constraint is data, not architectural capacity.",

 "Calibration is ordered inversely to accuracy. ViT-Base is best calibrated despite being least accurate, while both EfficientNets are worst calibrated despite outperforming it: a model may be more frequently wrong yet more honest about its uncertainty [guo2017, naeini2015]. This bears directly on the deployed system, whose abstention depends on confidence being meaningful, so selecting on accuracy alone would have degraded abstention without appearing in any accuracy metric.",

 "One limitation qualifies the architectural claim. All models were trained at a single learning rate selected for the ResNet-50 baseline, and ViT-Base is conventionally fine-tuned at a lower rate, so part of its deficit is plausibly hyperparameter mismatch. The defensible claim is narrower than architectural inferiority: under a protocol tuned to the convolutional baseline, ViT-Base underperforms substantially on this corpus.",

 "4.3 RQ1: does classification perform equitably across tone groups?",

 ('TABLE', 'Disaggregated classifier performance by validated tone band, with bootstrapped 95% confidence intervals over 2,000 resamples.', [
   ["Group", "n", "Accuracy [95% CI]", "Macro-F1 [95% CI]", "ECE"],
   ["Light", "538", "0.762 [0.723, 0.797]", "0.739 [0.696, 0.780]", "0.120"],
   ["Medium", "724", "0.782 [0.751, 0.812]", "0.735 [0.684, 0.779]", "0.092"],
 ]),

 "[[FIG:armB_fair_accuracy.png|Classifier accuracy by tone band with 95% bootstrap confidence intervals. The intervals overlap substantially, and the point estimates differ by two percentage points in favour of the larger group.]]",

 "The accuracy gap between Light and Medium is minus 0.020, with a 95% confidence interval of minus 0.070 to plus 0.026. The interval straddles zero, so no accuracy disparity is demonstrated between the two auditable bands. Consistent with the pre-committed protocol this is reported as a null rather than reframed, and the interval width is itself informative: at these group sizes the study can exclude disparities larger than roughly seven percentage points, but not smaller ones.",

 "[[FIG:armB_fair_ece.png|Expected calibration error by tone band. Accuracy is equivalent across the two groups while calibration is not.]]",

 "The substantive finding lies in calibration rather than accuracy. Expected calibration error is approximately 30% higher for the Light group, 0.120 against 0.092: the model is more confidently wrong on the smaller of the two groups. Accuracy parity therefore conceals a difference in the reliability of the confidence signal, which is precisely the quantity a deployed system uses to decide whether to advise at all. This is a direct argument for reporting calibration alongside accuracy in fairness audits, and it motivates the abstention analysis in Section 4.10.",

 "Per-class recall exhibits gaps in both directions, with the Light group weaker at grade 1 and stronger at grade 3. These contrasts rest on small per-cell counts and are not accompanied by intervals; they are reported as observations warranting further investigation rather than as findings.",

 "4.4 RQ2: does classifier bias propagate into recommendations?",

 "The two-pass design is described in Section 3.6: every subject is passed through the same deterministic recommender on the dermatologist's grade and on the model's prediction, so that any divergence between the resulting lists is attributable to the classifier alone.",

 ('TABLE', 'Recommendation outcomes conditioned on upstream classifier correctness.', [
   ["Outcome", "n", "List overlap (Jaccard)", "Relevance P@10", "Price shift (USD)"],
   ["Correctly classified", "1,116", "1.000 [1.000, 1.000]", "0.995 [0.993, 0.997]", "0.00"],
   ["Misclassified", "341", "0.515 [0.495, 0.536]", "0.987 [0.983, 0.991]", "-0.73 [-2.07, +0.60]"],
 ]),

 "[[FIG:rq2_cost_of_error.png|List overlap between the recommendations a subject received and those they should have received, conditioned on whether the classifier was correct.]]",

 "The overlap penalty of misclassification is 0.485, with an interval of 0.465 to 0.504. A misclassified user does not receive a marginally worse list; they receive one sharing barely half its contents with the list they should have received. On a corpus where 23.4% of subjects are misclassified, this is the harm the system realises at the point of recommendation rather than at the point of classification. Propagation is real and substantial, and this is the principal measurement the project set out to make.",

 ('TABLE', 'Recommendation outcomes disaggregated by tone band.', [
   ["Band", "n", "Accuracy", "Jaccard", "Relevance P@10", "Missed referrals"],
   ["Light", "538", "0.762", "0.885 [0.864, 0.903]", "0.989 [0.986, 0.993]", "2.2%"],
   ["Medium", "724", "0.782", "0.894 [0.878, 0.909]", "0.996 [0.995, 0.998]", "4.0%"],
 ]),

 ('TABLE', 'Light minus Medium gaps in recommendation outcomes, with 95% confidence intervals.', [
   ["Measure", "Gap", "95% CI", "Excludes zero?"],
   ["List overlap (Jaccard)", "-0.010", "[-0.035, +0.015]", "No"],
   ["Relevance P@10", "-0.007", "[-0.011, -0.004]", "Yes, but immaterial"],
   ["Price (USD)", "-0.29", "[-0.92, +0.36]", "No"],
   ["Missed referrals", "-1.8 pp", "[-3.7, 0.0]", "No"],
 ]),

 "[[FIG:rq2_jaccard_by_tone.png|Recommendation fidelity by tone band: overlap between the list a subject received and the list they should have received, with 95% confidence intervals. The two bands are indistinguishable.]]",

 "[[FIG:rq2_price_shift.png|Distribution of the price change caused by misclassification, by tone band. Both distributions are centred on zero and closely superimposed; the tails are sparse and symmetric.]]",

 "The tone-differential picture is null. Only the relevance gap excludes zero, and at 0.007 it falls far below any threshold at which the products a user sees would change. Reporting statistical resolution and practical magnitude separately, no tone-differential propagation is demonstrated. With two auditable bands, no dark-skin population, and an upstream classifier that itself showed no accuracy disparity, this arm cannot settle the question; the interval widths document that limitation rather than obscuring it.",

 "Two secondary observations stand. Across all simulated users only about 100 distinct products, roughly 6% of the eligible catalogue, are ever surfaced, the concentration effect the recommender literature documents [abdollahpouri2019, ge2010]. And the pipeline beats a trivial baseline, with mean precision at 10 of 0.993 against 0.425 for the popularity-only ranker, confirming the mapping table contributes beyond surfacing best-sellers.",

 "A limitation bears on the metric itself. Precision at k is close to saturated in both arms at approximately 0.99, because the mapping table endorses overlapping ingredient classes at adjacent grades, so a one-grade error still returns products carrying an endorsed class. It is therefore an insensitive instrument for propagation, and list overlap is the informative measure here.",

 "4.5 RQ3: which mitigation strategies reduce disparity?",

 "Six strategies were compared across three seeds, eighteen runs in total. Given the RQ1 null the experiment was scoped in advance to the two deficits the audit did find: the calibration gap and severe-class weakness.",

 ('TABLE', 'Mitigation strategies compared across three seeds.', [
   ["Strategy", "Macro-F1", "Severe F1", "Accuracy gap", "ECE gap", "Worst-group accuracy"],
   ["Group reweighting", "0.743 +/- 0.035", "0.595", "0.034", "0.051", "0.760"],
   ["ERM (baseline)", "0.735 +/- 0.031", "0.591", "0.036", "0.027", "0.756"],
   ["Class reweighting", "0.731 +/- 0.026", "0.582", "0.052", "0.038", "0.730"],
   ["Oversampling", "0.731 +/- 0.035", "0.570", "0.082", "0.034", "0.723"],
   ["GroupDRO", "0.726 +/- 0.011", "0.531", "0.033", "0.014", "0.752"],
   ["Focal loss", "0.708 +/- 0.036", "0.578", "0.070", "0.071", "0.692"],
 ]),

 "[[FIG:rq3_tradeoff_curve.png|The six strategies on an accuracy-fairness plane. They form an overlapping cluster rather than a frontier.]]",

 "No strategy demonstrably improves on the unmitigated baseline: the best apparent gain is 0.008 against a seed standard deviation of plus or minus 0.035, roughly a quarter of the noise.",

 "The scale of that noise is the most transferable result here. Under empirical risk minimisation alone, macro-F1 across three seeds was 0.702, 0.740 and 0.765, a spread of 0.063 and eight times the largest apparent mitigation gain. Run at a single seed, almost any ranking of the six strategies could have been produced, and would have looked publishable. This is the failure mode Bouthillier et al. and Picard document [bouthillier2021, picard2021], observed here on a live experiment rather than in principle.",

 ('TABLE', 'Qualified observations from the mitigation comparison. None constitutes a demonstrated improvement over the baseline.', [
   ["Strategy", "Observation", "Interpretation"],
   ["GroupDRO [sagawa2020]", "Smallest calibration gap (0.014) and a third the seed variance of the others",
    "Interval includes the baseline, so no improvement is shown, but stability is a practical property deserving more seeds"],
   ["Focal loss [lin2017]", "Lowest macro-F1, worst worst-group accuracy, largest and most volatile calibration gap",
    "Down-weighting easy examples appears to destabilise training on a small, imbalanced corpus"],
   ["Oversampling", "Widened the accuracy gap to 0.082, the largest observed, without improving macro-F1",
    "Resampling rare cells with replacement on a corpus this small amplifies whichever few images occupy them"],
 ]),

 "With three seeds and a seed standard deviation near 0.035, this design resolves differences of roughly 0.05 or larger, and none reaches that. The correct reading is underpowered to distinguish, not shown to be equivalent.",

 "4.6 RQ4: is the model causally sensitive to skin tone?",

 "The audits above are correlational. This probe intervenes, rendering each image at five tone levels while holding lesion content fixed as described in Section 3.7.",

 ('TABLE', 'Counterfactual tone probe. Positive severity shift means the model assigns a more severe grade.', [
   ["Tone shift", "Flip rate [95% CI]", "Change in expected severity [95% CI]", "Change in confidence"],
   ["-20 deg (darker)", "6.8% [5.5, 8.2]", "+0.056 [+0.047, +0.066]", "-0.021"],
   ["-10 deg", "3.6% [2.6, 4.5]", "+0.019 [+0.013, +0.026]", "-0.010"],
   ["0 deg (control)", "0.0% [0.0, 0.0]", "+0.000", "+0.000"],
   ["+10 deg", "5.4% [4.2, 6.6]", "-0.005 [-0.014, +0.005]", "-0.022"],
   ["+20 deg (lighter)", "20.1% [17.8, 22.0]", "+0.104 [+0.082, +0.126]", "-0.095"],
 ]),

 "[[FIG:rq4_tone_sensitivity.png|Prediction change under controlled tone shift with lesion content held constant. The zero-shift control returns exactly zero and the darkening arm is dose-responsive.]]",

 "The answer is yes, and two features support the claim beyond a bare significance test. The zero-shift control returns exactly zero, confirming the rendering pipeline introduces no spurious variation. And the darkening arm is dose-responsive: flip rate rises from 3.6% to 6.8% and severity shift from 0.019 to 0.056 as the shift doubles, with both intervals excluding zero. A dose-response relationship is materially stronger evidence than a single significant contrast. Magnitudes are modest in absolute terms, 0.056 being roughly 1.9% of the four-point scale, but the behaviourally relevant figure is the flip rate: 6.8% of subjects receive a different grade, and therefore a different product set, purely because their skin was rendered darker.",

 "Lightening produced a larger effect than darkening, and two explanations were tested against the evidence below. Clipping is rejected; training-distribution sparsity accounts for it. The model is not merely wrong at the light extreme, it is uncertain there, the signature of operating beyond the density of its training data.",

 ('TABLE', 'Two candidate explanations for the asymmetry between the lightening and darkening arms.', [
   ["Explanation", "Prediction if true", "Measured", "Verdict"],
   ["Luminance clipping destroys texture",
    "Substantial pixel saturation at +20 deg",
    "0.19% of skin pixels clip at +20 deg; 0.00% at -20 deg",
    "Rejected"],
   ["Shift moves images beyond training density",
    "More images in sparse tails, and lower confidence, when lightening",
    "33.1% in sparse tails at +20 deg against 21.5% at -20 deg; confidence -0.095 against -0.021",
    "Supported"],
 ]),

 "This is the finding worth carrying forward. Instability is greatest precisely where training coverage is thinnest, which is the mechanism by which under-representation becomes unreliable behaviour for the under-represented, demonstrated causally rather than inferred from an accuracy table.",

 "The probe manipulates a rendered proxy for tone, not the covariates accompanying it in real subjects, and holds yellow-blue chromaticity fixed while solving for lightness whereas real tone variation moves both. These are reasons to weight the ten-degree and minus-twenty-degree results above the plus-twenty result, not to discount the dose-response finding.",

 "4.7 RQ5: is the review corpus representationally biased?",

 "Of 1,094,411 reviews, 923,802 carry a usable tone band.",

 ('TABLE', 'Review representation by reviewer-reported skin tone.', [
   ["Band", "Reviews", "Share [95% CI]", "Mean rating", "Median price (USD)"],
   ["Light", "728,833", "78.9% [78.8, 79.0]", "4.294 [4.292, 4.297]", "40"],
   ["Medium", "168,350", "18.2% [18.1, 18.3]", "4.303 [4.297, 4.308]", "39"],
   ["Deep", "26,619", "2.9% [2.8, 2.9]", "4.267 [4.252, 4.280]", "38"],
 ]),

 "[[FIG:rq5_representation.png|Review volume by reviewer-reported skin tone. The ratio between the lightest and deepest bands is 27.4 to 1.]]",

 "The representation ratio is 27.4 to 1, Light to Deep. The aggregate review signal any review-driven recommender depends on is overwhelmingly a lighter-skin signal, a bias in the corpus itself prior to any model being trained on it.",

 "[[FIG:rq5_rating_ci.png|Mean star rating by reviewer tone band with 95% bootstrap confidence intervals. Note the vertical scale: the entire range spans 0.04 of a star.]]",

 "[[FIG:rq5_price_ci.png|Median price of reviewed products by reviewer tone band with 95% bootstrap confidence intervals.]]",

 "Differences in rating and price are statistically significant, inevitably so at this sample size, but negligible in effect size, with epsilon squared of 0.0001 and 0.0004. The rating figure makes the point better than the statistic does: the vertical axis spans four hundredths of a star, and the Deep band's interval is visibly wider because it rests on 2.9% of the data. Reporting effect sizes alongside significance [cohen1988] is what prevents these being oversold as evidence of differential experience.",

 ('TABLE', 'Reviews mentioning post-inflammatory hyperpigmentation and related concerns.', [
   ["Band", "Mentions", "Rate [95% CI]"],
   ["Light", "20,748", "2.85% [2.81, 2.89]"],
   ["Medium", "7,726", "4.59% [4.48, 4.70]"],
   ["Deep", "1,794", "6.74% [6.45, 7.04]"],
 ]),

 "[[FIG:rq5_pih_coverage.png|Rate at which reviewers in each tone band raise hyperpigmentation, dark-mark and scarring concerns.]]",

 "The substantive finding is in concern coverage. Reviewers with deeper skin tones raise hyperpigmentation, dark-mark and acne-scarring concerns at 2.4 times the rate of lighter-toned reviewers, with a chi-squared statistic of 2347.5 and Cramer's V of 0.050. The two findings compound: the concern most characteristic of darker skin, and clinically documented as such [davis2010], is raised disproportionately often by a group contributing under 3% of the corpus. A recommender learning product-concern associations from review text has correspondingly thin evidence precisely where the concern is most prevalent.",

 "4.8 The text pipeline",

 "The sentiment classifier described in Section 3.8 achieves accuracy 0.974, macro-F1 0.974 and ROC-AUC 0.996 on a held-out set of 24,000 reviews. It exists to test whether bias enters through the text route as well as the image route: if the model read one group's reviews less accurately, that group's expressed preferences would be measured less reliably.",

 ('TABLE', 'Sentiment classifier accuracy disaggregated by reviewer tone band.', [
   ["Band", "n", "Accuracy [95% CI]"],
   ["Light", "15,887", "0.9751 [0.9726, 0.9773]"],
   ["Medium", "3,630", "0.9774 [0.9725, 0.9821]"],
   ["Deep", "575", "0.9704 [0.9565, 0.9827]"],
 ]),

 "[[FIG:sentiment_by_tone.png|Sentiment classifier accuracy by reviewer tone band with 95% confidence intervals. The intervals overlap heavily, and the deepest band's is roughly five times wider.]]",

 "Accuracy spread is 0.0070 with heavily overlapping intervals, so no disparity is demonstrated: the text pipeline reads all three bands with effectively equal accuracy. The qualification is instructive rather than incidental. The Deep band contributes only 575 test reviews, giving an interval roughly five times wider than the Light band's. The sample sizes reproduce the corpus imbalance documented above: even where a downstream model is even-handed, the data it learns from is not, and it is that imbalance which limits what can be established about the smallest group.",

 "A retrospective validation follows. Across 2,255 products, text-derived and rating-derived positive rates correlate at 0.904, so roughly 82% of product-level variance is shared. The rating proxy relied upon in the propagation and corpus analyses was therefore a sound stand-in. One caution applies: because the labels are the rating threshold, agreement between model and rating equals accuracy by construction, and the complement cannot be read as evidence the rating mislabelled those reviews. The product-level correlation is the finding that survives this objection.",

 "4.9 RQ6: cross-arm replication",

 "RQ6 was not completed. It requires the reference audit on a corpus carrying ground-truth Fitzpatrick labels, and no copy pairing that corpus's images with its tone metadata could be obtained. The code for that arm is implemented and hardened but unexecuted.",

 ('TABLE', 'Sources attempted for a tone-labelled dermatology corpus, and the outcome of each.', [
   ["Source type", "Outcome"],
   ["Public mirror, images only", "16,574 images present, no tone labels"],
   ["Public mirror, second copy", "Stub containing 13 images"],
   ["Third-party redistribution", "Repository deleted, returns not found"],
   ["Fourth-party copy", "Could not be attached successfully"],
   ["Curated competition release", "Requires an invitation"],
   ["Original author repository", "Manifest of image URLs; many no longer resolve"],
 ]),

 "Following the gate outcome in Section 4.1, that arm had become the primary powered fairness audit rather than a supporting one, since it alone offered ground-truth labels and a population spanning Fitzpatrick I to VI. Its absence means the fairness findings here rest entirely on a corpus with a demonstrably compressed tone range, and the nulls in RQ1 and RQ2 must be read in that light: they establish that no disparity was detectable between two adjacent light-to-medium bands, not that none exists across the full range of human skin tone.",

 "That limitation is itself evidence for the project's broader argument. Wen et al. document that most dermatological datasets do not record skin tone [wen2022]; this project adds that even where tone labels exist, obtaining them attached to their images may not be possible, which bounds what independent replication of fairness results can achieve.",

 "4.10 The deployed system",

 "The derived abstention threshold sits at 0.85, giving coverage 0.677 and selective accuracy 0.851, an improvement of 8.5 percentage points over the full-coverage baseline, purchased by declining to advise on a third of cases. Because abstention is itself an allocation it was checked per group: coverage is 0.697 for Light against 0.689 for Medium, a gap of 0.008. Had it fallen materially harder on one group, the system would have been quietly denying service to it, an allocative harm invisible to accuracy metrics.",

 "Refusal behaviour achieves a refusal rate of 1.00 and a false-refusal rate of 0.00 across 21 adversarial medical requests and 12 benign controls. That result required correction: the first evaluation returned 0.81, exposing four genuine gaps, a missed plural, a reversed word order, an unanticipated phrasing and a pronoun variant. The rules were widened and the set re-run. That the set located real defects on first contact is the strongest available evidence it constitutes a test rather than a demonstration; a set passing immediately would have evidenced only that it was written to match the implementation. Transcripts are in Appendix D.",

 "4.11 Synthesis",

 ('TABLE', 'Summary of findings.', [
   ["#", "Finding"],
   ["1", "ACNE04 contains no auditable dark-skinned population; the apparent one is a lighting and inflammation artefact, robust across three estimators."],
   ["2", "ResNet-50 achieves 0.766 accuracy and 0.746 macro-F1, outperforming EfficientNet-B0/B3 and ViT-Base by more than seed noise."],
   ["3", "No accuracy disparity is demonstrated between Light and Medium bands, but calibration is about 30% worse for the smaller group."],
   ["4", "Misclassification propagates strongly into recommendations: list overlap falls from 1.000 to 0.515."],
   ["5", "That propagation is not demonstrably tone-differential on this corpus."],
   ["6", "No mitigation strategy beats the unmitigated baseline; seed variance exceeds every between-strategy difference eightfold."],
   ["7", "The model is causally tone-sensitive: 6.8% of predictions flip under a 20-degree darkening shift, dose-responsively, with lesion content fixed."],
   ["8", "Model instability is greatest where training tone coverage is sparsest."],
   ["9", "The review corpus is representationally biased 27 to 1, while deeper-toned reviewers raise hyperpigmentation concerns at 2.4 times the rate."],
   ["10", "Refusal behaviour reaches 1.00 refusal and 0.00 false refusal after correction of four defects located by the adversarial set."],
 ]),

 "The central result is the conjunction of findings 3, 5 and 7. A model that shows no disparity in disaggregated accuracy, and whose errors produce no demonstrably tone-differential downstream harm, is nevertheless demonstrably sensitive to skin tone at the mechanism level. The two classes of evidence are not in conflict, because they answer different questions. Aggregate audits average over a population whose tone range is narrow, and can only detect what that population makes visible; a causal probe manufactures the contrast directly and does not depend on that population existing.",

 "The practical implication is uncomfortable and is the project's principal claim. An organisation auditing a system of this kind by the standard method, disaggregated accuracy across available tone groups, would have found nothing and concluded the system was tone-blind. The causal probe shows it is not. Reporting only the former can therefore license a false assurance, and the two methods should be treated as complements rather than as alternatives.",
]

CH4_RELATED = [
 "The findings can now be positioned against the work reviewed in Chapter 2.",

 "On dermatological fairness, the result here is partly discordant and the reason is instructive. That literature consistently reports accuracy gaps across tone groups [daneshjou2022, kinyanjui2020, groh2021]; this project does not. The discordance is not evidence against those findings but follows from the corpus: those studies audit populations spanning Fitzpatrick I to VI, whereas the gate in Section 4.1 established that this one spans two adjacent light-to-medium bands, and a null across two neighbouring bands does not contradict a gap measured across six. What this project adds is that the same model returning that null is causally tone-sensitive when probed directly. The caution generalises: a disaggregated audit is bounded by the diversity of the population it audits, and reporting one without characterising that bound overstates what has been established.",

 "On pipeline fairness, the propagation measurement supports the composition literature's central claim in a new domain. Dwork and Ilvento argue fairness properties do not survive composition and must be assessed on the composed system [dwork2019]; the measurement here shows why that matters concretely, since a classifier error costing 23.4% accuracy translates into a list sharing barely half its contents with the correct one. That magnitude is not readable from the classifier metric alone. The catalogue concentration observed, roughly 6% of eligible products ever surfaced, reproduces the exposure concentration documented in the popularity-bias literature [abdollahpouri2019, ge2010] in a system using no popularity signal in its primary scoring term, suggesting concentration arises from the narrowness of the endorsed ingredient classes rather than from a feedback loop.",

 "On skincare recommendation, the closest comparator is Lee et al. [lee2024]. Their reported imbalance in product concern labelling is independently reproduced here on a different catalogue, 19.5% of products carrying an acne-directed active against 55.3% carrying an anti-ageing or hydration active; that two independently assembled catalogues show the same skew suggests a property of the market rather than of either dataset. And where that work treats bias as a limitation for future data collection to resolve, the measurement here suggests the expectation is optimistic, since the 27 to 1 representation ratio in the review signal would not be addressed by broader image collection alone.",

 "Finally, the mitigation null aligns with the benchmarking literature rather than the mitigation literature. Bouthillier et al. and Picard argue reported improvements frequently fall within seed variance [bouthillier2021, picard2021]; this project provides a worked instance, six strategies separated by less than an eighth of the variance a single strategy exhibits across three seeds. Fairness mitigation results reported at a single seed should therefore be treated as uninformative, and the noise floor characterised before any difference is claimed. That the same test resolves the architecture comparison while returning unresolved for mitigation is evidence it discriminates rather than merely declines to conclude.",
]

# --- Chapter 5: Conclusion ------------------------------------------------

CH5_CONC = [
 "This project asked whether disparities in automated skin assessment propagate into the product recommendations that depend on them. It built a two-stage pipeline of the kind deployed commercially, an image-based acne severity classifier feeding a deterministic content-based recommender over a large review corpus, and measured fairness at both stages and across the boundary between them.",

 "The propagation measurement is the principal contribution and it is unambiguous. Passing every subject through the same deterministic recommender on the true grade and on the predicted grade, a misclassified user receives a list sharing 0.515 of its contents with the list they should have received, against 1.000 when correctly classified. Because every component downstream of the classifier is deterministic, that divergence is attributable to the classifier alone. Reported at the classifier, the system's error rate is 23.4%; reported where the user experiences it, the same error costs half the recommendation list. Fairness work stopping at the first number understates the second, and the two are not recoverable from one another.",

 "The second contribution is methodological and emerged from a conjunction the project did not set out to find. The disaggregated audit found no accuracy disparity between the two auditable bands. The propagation analysis found no tone-differential downstream harm. The counterfactual probe, on the same model, found 6.8% of predictions flip under a twenty-degree darkening shift with lesion content fixed, dose-responsively, with a zero-shift control returning exactly zero. A model can therefore be demonstrably tone-sensitive at the mechanism level while showing no disparity in aggregate accuracy or downstream outcomes. These results are not in tension: aggregate audits average over a population whose tone range is narrow and detect only what that population makes visible, whereas a causal probe manufactures the contrast directly and does not require that population to exist. An organisation auditing such a system by the standard method would have found nothing and concluded it was tone-blind.",

 "A third contribution is negative and concerns measurement discipline. No mitigation strategy outperformed the baseline, and the reason matters more than the result: a single strategy across three seeds spanned a macro-F1 range eight times larger than the best apparent gain between strategies. Single-seed mitigation comparisons on corpora of this size are not weak evidence but uninformative.",

 "Two corpus findings stand independently of the models. The image corpus contains no auditable population of dark-skinned subjects, robust across three tone estimators and established through visual validation after the first returned an implausible result. And the review corpus is representationally biased 27.4 to 1, while the concern most characteristic of darker skin is raised at 2.4 times the rate by the group contributing under 3% of reviews, so a recommender learning from it has the thinnest evidence precisely where that concern is most prevalent. Both are reported as results rather than as obstacles.",

 "The work's principal limitation is inseparable from its setting. The reference audit on a corpus carrying ground-truth Fitzpatrick labels could not be conducted, no copy pairing that corpus's images with its tone metadata having been obtainable. All fairness findings therefore rest on a corpus with a compressed tone range, and the nulls must be read accordingly: they establish that no disparity was detectable between two adjacent light-to-medium bands, not that none exists across the range of human skin tone. Stating this plainly is what makes the causal probe's positive finding load-bearing, since it is the one result that does not depend on the missing population.",
]

CH5_FUTURE = [
 "Four directions follow directly from the limitations recorded above, ordered by the value each would add.",

 "The first is completing the reference arm. The training and audit code is implemented and hardened, asserting the presence and schema of tone metadata and refusing to proceed below a minimum matched-image count; it needs only a corpus pairing images with ground-truth Fitzpatrick labels. That would supply the powered audit across Fitzpatrick I to VI this project could not conduct, and would answer RQ6 directly by testing whether findings established on a large tone-labelled corpus replicate on a small tone-estimated one. Given the acquisition difficulties documented in Section 4.9, an institutional data-sharing agreement is a more realistic route than public download, and the effort is better placed there than in further modelling.",

 "The second is a stricter relevance standard for the propagation analysis. Precision at k saturated near 0.99 in both arms because the mapping table endorses overlapping ingredient classes at adjacent severity grades, making it an insensitive instrument. A grade-specific reference standard, endorsing a narrower class set at each grade, would make relevance informative alongside list overlap and would likely reveal propagation effects the current metric cannot resolve. This is the cheapest of the four and would strengthen the project's principal measurement.",

 "The third concerns the architecture comparison, which trained every model at a single learning rate selected for the convolutional baseline. A per-architecture learning-rate sweep would establish whether the Vision Transformer's substantial deficit reflects architecture or hyperparameter mismatch. The current claim is deliberately narrow, and this would either widen it legitimately or retract it.",

 "The fourth follows the one lead the mitigation experiment produced. GroupDRO returned the smallest calibration gap and roughly a third the seed variance of the other strategies. Its interval includes the baseline, so no improvement is demonstrated, but low variance is a practical property in its own right, and given that the experiment resolves differences only above roughly 0.05, more seeds would establish whether the stability is real. The same design should be repeated at ten seeds rather than three before any strategy is recommended.",

 "Beyond these, two extensions would broaden the contribution. The counterfactual probe currently shifts lightness while holding yellow-blue chromaticity fixed, whereas real tone variation moves both; a probe traversing a physically plausible manifold of skin colour would strengthen the causal claim. And the propagation methodology is not specific to skincare. Any pipeline in which a classifier's output selects what a user is subsequently shown, including diagnostic triage and content moderation, admits the same two-pass measurement, and the finding that aggregate parity can coexist with mechanism-level sensitivity is unlikely to be confined to this domain.",
]

CH5_REFLECT = [
 "The most instructive episode in the project was a measurement failure that was nearly accepted as a finding.",

 "The first implementation of the tone estimator returned a distribution placing 59.5% of the image corpus in the darkest band. That number was not obviously wrong in isolation: it was produced by a documented method from the dermatological literature, implemented without error, and it would have supported an unusually well-powered fairness audit across three tone bands. It was also, on reflection, not credible, since the corpus is predominantly Chinese. The proposal had committed to validating tone estimates by manual inspection of a stratified sample, and rendering montages of real faces grouped by estimated band made the failure visible within minutes. The darkest band contained ordinary light-to-medium skinned faces photographed in shadow, and the lightest band was dominated by tightly cropped, heavily inflamed images whose redness had inflated the estimate.",

 "The second error mode was the more serious of the two, and it is worth stating why. Inflammation was pushing severe cases systematically toward one tone band. Had the audit proceeded on those labels, it would have found a tone disparity that was in fact a severity artefact, and every subsequent analysis, propagation, mitigation and the causal probe, would have been built on it. The project would have produced a confident and entirely spurious positive finding. Correcting the estimator, by discarding the reddest quartile of skin pixels and retaining the brighter half of the remainder, moved the median to 37.3 degrees, consistent with Fitzpatrick type III and with the corpus demographic.",

 "The correction was expensive. It triggered the pre-committed decision gate, restricted all tone analysis to two bands, reduced the applied arm from a primary audit to an under-powered replication, and turned three research questions into null results. A validation step designed as a checkbox instead reshaped the project. The lesson taken is that the value of a validation procedure is realised precisely when it contradicts a convenient result, and that committing to it in advance, before knowing what it would cost, is what made it possible to act on.",

 "A related discipline proved its worth twice more. In the counterfactual probe, the obvious explanation for the asymmetry between lightening and darkening was luminance clipping. Measuring it directly rejected the hypothesis, since only 0.19% of pixels clipped, and pursuing the alternative produced the more interesting explanation of training-distribution sparsity and, with it, the causal link between under-representation and instability that became a central finding. In the safety evaluation, the adversarial set failed on first contact at a refusal rate of 0.81, exposing four real defects. A set that had passed immediately would have evidenced only that it was written to match the implementation.",

 "Two things would be done differently. The tone estimator would be validated visually before any downstream code was written rather than after, since the correction invalidated work already completed. And the reference corpus would have been acquired and verified in the first week rather than assumed available, because by the time its inaccessibility was established, the schedule no longer permitted an alternative. That failure is documented in Section 4.9 rather than minimised, and the six attempted sources are recorded so the obstacle is reproducible by anyone who follows.",

 "What the project ultimately demonstrates is narrower than it set out to prove and more useful than a confirmed hypothesis would have been. Three of its research questions returned nulls. The proposal had committed in advance to treating a rigorous null with intervals demonstrating power as a valid outcome, and that commitment held when it became inconvenient. The conjunction of those nulls with the causal probe's positive result is the contribution, and it exists only because none of the three was quietly reframed.",
]

# --- Appendices -----------------------------------------------------------

APPENDICES = {
 "Appendix A: Project Proposal": [
   "The approved project proposal is submitted separately.",
   "The proposal committed to six research questions, a two-arm fairness design, an early go/no-go decision gate on the image corpus's tone range, a pre-committed metric set fixed before results were inspected, and a demonstrator whose recommendation logic sits entirely outside any generative component. Five of the six research questions were completed. The exceptions and deviations are recorded below so that the delivered work can be compared against what was promised.",
   "The decision gate was invoked. Section 4.1 records the outcome as no-go for the primary path, which triggered the contingency specified in the proposal: analytical weight shifted toward the review corpus audit, and the compressed tone range was reported as a quantified finding rather than absorbed as an obstacle.",
   "The reference arm on a tone-labelled dermatology corpus was not executed, no copy pairing its images with its tone metadata having proved obtainable. RQ6, which depends on it, is therefore unanswered. Section 4.9 documents the six sources attempted and the consequences for the interpretation of the remaining results.",
   "One deviation strengthened a proposal commitment rather than relaxing it. The proposal permitted a language model to phrase questions, parse free-text replies and verbalise output, subject to a strict prohibition on influencing recommendations. The delivered demonstrator contains no language model at all, making that separation structural rather than behavioural.",
 ],

 "Appendix B: Project Management": [
   "Work proceeded in the phases set out in the proposal, with two departures worth recording.",
   "The first phase covered acquisition and cleaning of all three corpora, computation of the tone estimate, and the decision gate. It overran, because the first tone estimator produced an implausible distribution and the visual validation that identified the cause also invalidated work already built on it. The correction is described in Chapter 3 and reflected on in Chapter 5. The overrun was accepted deliberately, on the grounds that every downstream analysis depended on the tone labels being correct.",
   "Subsequent phases covered exploratory analysis, the mapping table, classifier training, the fairness audit, the counterfactual probe, the mitigation comparison, the architecture comparison, review mining and the corpus audit, the propagation analysis, and finally the demonstrator with its adversarial testing. The demonstrator was built last and timeboxed, per the proposal's risk register, so that it could not displace research work.",
   "The second departure was the reference arm. Its code was written and hardened early, on the assumption that the corpus would be obtainable. Repeated acquisition attempts failed across six sources, and by the point that was established, the schedule no longer permitted an alternative design. In retrospect the corpus should have been acquired and verified before any code depending on it was written. This is recorded as a project management failure rather than as an external obstacle, since the risk was foreseeable.",
   "Model training was performed on hosted GPU infrastructure, the local machine having no CUDA device. Training notebooks are held as version-controlled Python sources and generated into notebook form for upload, so experiment code remains reviewable and diffable. Each carries a smoke-test flag permitting a short validation run before a full commit, a precaution adopted after an interactive session's outputs were lost.",
 ],

 "Appendix C: Artefact/Dataset": [
   "Three public datasets were used. ACNE04 supplies 1,457 dermatologist-graded facial images under the Hayashi criterion and is described in the ICCV 2019 paper cited in the references. The Sephora products and skincare reviews dataset supplies 8,494 products and 1,094,411 reviews and is publicly available on Kaggle. The Fitzpatrick17k corpus was sought for the reference arm and could not be obtained in a form pairing its images with its tone metadata; Section 4.9 records the sources attempted.",
   "All code is submitted in the accompanying code folder. It comprises the data preparation and analysis scripts, the training notebooks, the deterministic recommender module, the demonstrator application, the generated reports containing every statistic quoted in this document, and the figures reproduced here. Each analysis script writes a corresponding report file, so every number in Chapter 4 is traceable to the script that produced it.",
   "The demonstrator is a Streamlit application providing image upload, severity assessment with abstention below the derived confidence threshold, conversational elicitation of skin type, sensitivity, dark-mark concern and budget, and explained product recommendations. It runs locally and retains no uploaded image.",
   "The trained classifier checkpoint is included. Reproducing the analyses requires the two public datasets, which are not redistributed here.",
 ],

 "Appendix D: Screencast": [
   "A screencast demonstrating the artefact and walking through the implementation is submitted separately.",
   "The adversarial prompt set and resulting transcripts are included in the submitted code folder as evidence of tested refusal behaviour. The set comprises 21 medical requests spanning direct, indirect, role-play, smuggled, procedural and emergency framings, alongside 12 benign cosmetic controls included to measure over-refusal. The final evaluation returns a refusal rate of 1.00 and a false-refusal rate of 0.00.",
   "The first evaluation returned 0.81 and is retained in the submission alongside the final run, because the four defects it exposed are more informative than the passing result. They were: a missed plural form, where a rule matching a singular medication name failed on its plural; a reversed word order, where a symptom description placing the subject after the verb escaped a rule anticipating the opposite; a phrasing outside the anticipated template, where a request for guidance not framed as a question was not recognised; and a pronoun variant. All four are ordinary linguistic variation rather than adversarial ingenuity, which is the point: a rule-based refusal layer fails on paraphrase, and only a set written independently of the implementation will reveal it.",
 ],
}
