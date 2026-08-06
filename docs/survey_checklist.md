# 调研清单（唯一权威视图）

**由 `python scripts/survey_checklist.py` 生成，不要手抄。**
论文状态只在 `docs/papers.tsv` 里声明；本文件只是渲染，且**每条声明都与别处的证据对过**。

- 共 **129** 条；已复现 **18** 条
- 对比学习一类 **8** 条（导师指定方向，见 `decisions.md` 2026-08-04）
- **尚未核实作者代码 101** 条——空的「作者代码」列意思是**没查过**，
  不是「没有代码」。这两件事必须分开数，否则"没查"会慢慢被读成"没有"。

**「作者代码」一列的链接一律取自论文正文**，不取会议页面元数据——后者不显示正文里的代码链接，
据此判断已误判过四篇（规矩见 `survey.md`「核实规矩」）。

**一致性检查通过**：无声明与证据冲突。

## ✅ 已复现（18）

| key | 出处 | 类别 | 标题 | 作者代码 | PDF |
|---|---|---|---|---|---|
| `cenet` | — | — | Cross-Modal Enhancement Network for Multimodal Sentiment Analysis | [链接](https://github.com/Say2L/CENet) | — |
| `mctn` | — | — | Found in Translation: Learning Robust Joint Representations by Cyclic  | [链接](https://github.com/hainow/MCTN) | ✓ |
| `mfm` | — | — | Learning Factorized Multimodal Representations | [链接](https://github.com/pliang279/factorized/) | ✓ |
| `tetfn` | — | — | TETFN: A text enhanced transformer fusion network for multimodal senti | — | — |
| `self_mm` | AAAI 2021 | — | Learning Modality-Specific Representations with Self-Supervised Multi- | [链接](https://github.com/thuiar/Self-MM) | — |
| `mfn` | AAAI-18 | — | Memory Fusion Network for Multi-view Sequential Learning | [链接](https://github.com/pliang279/MFN) | ✓ |
| `graph_mfn` | ACL 2018 | — | Multimodal Language Analysis in the Wild: **CMU-MOSEI Dataset** and In | — | — |
| `lmf` | ACL 2018 | — | Efficient Low-rank Multimodal Fusion With Modality-Specific Factors | [链接](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) | ✓ |
| `mult` | ACL 2019 | — | Multimodal Transformer for Unaligned Multimodal Language Sequences | [链接](https://github.com/yaohungt/Multimodal-Transformer) | ✓ |
| `bert_mag` | ACL 2020 | — | Integrating Multimodal Information in Large Pretrained Transformers | [链接](https://github.com/WasifurRahman/BERT_multimodal_transformer) | — |
| `confede` | ACL 2023 长文, pp.7617-7630 | contrastive | ConFEDE | [链接](https://github.com/XpastaX/ConFEDE/) | — |
| `dmd` | CVPR 2023 (highlight), pp.6631-6640 | — | DMD | [链接](https://github.com/mdswyz/DMD) | — |
| `tfn` | EMNLP 2017 | — | Tensor Fusion Network for Multimodal Sentiment Analysis | — | ✓ |
| `mmim` | EMNLP 2021 | contrastive | Improving Multimodal Fusion with Hierarchical Mutual Information Maxim | [链接](https://github.com/declare-lab/Multimodal-Infomax) | ✓ |
| `almt` | EMNLP 2023 | — | Learning Language-guided Adaptive Hyper-modality Representation for Mu | — | ✓ |
| `dpdf_lq` | EMNLP 2025 主会 | — | DPDF-LQ | [链接](https://github.com/ZhouMiaoGX/DPDF-LQ) | — |
| `misa` | arXiv:2005.03545 | — | MISA: Modality-Invariant and -Specific Representations for Multimodal  | [链接](https://github.com/declare-lab/MISA) | ✓ |
| `clgsi` | naacl2024f | contrastive | CLGSI: A Multimodal Sentiment Analysis Framework based on Contrastive  | [链接](https://github.com/AZYoung233/CLGSI) | — |

## ▶ 计划中（2）

| key | 出处 | 类别 | 标题 | 作者代码 | PDF |
|---|---|---|---|---|---|
| `kuda**_knowledge_guided_dynamic_modality_attention_fusion` | Findings of EMNLP 2024 | — | KuDA** Knowledge-Guided Dynamic Modality Attention Fusion | [链接](https://github.com/MKMaS-GUET/KuDA) | — |
| `feada` | IJCNLP-AACL 2025 | — | FeaDA | [链接](https://github.com/PowerLittleYin/FeaDA-main) | — |

## ◻ 候选（103）

| key | 出处 | 类别 | 标题 | 作者代码 | PDF |
|---|---|---|---|---|---|
| `qa_moe**_quality_aware_mixture_of_experts,_robust_msa` | ACL 2026 长文 | — | QA-MoE** Quality-Aware Mixture of Experts, Robust MSA | — | — |
| `active_perceptual_inference_a_corticotha` | CVPR2026 | — | Active Perceptual Inference: A Corticothalamic-Inspired Dynamic Nested | — | — |
| `cica_coupling_confidence_aware_pretraini` | CVPR2026 | — | CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed A | — | — |
| `conflict_aware_adaptive_cross_reconstruc` | CVPR2026 | — | Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment  | — | — |
| `factorize_reconstruct_enhance_a_unified` | CVPR2026 | — | Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Se | — | — |
| `multi_metric_representation_learning_str` | CVPR2026 | — | Multi-Metric Representation Learning Strategy Based on Clustering for  | — | — |
| `prototype_as_prompt_multimodal_sentiment` | CVPR2026 | — | Prototype-as-Prompt: Multimodal Sentiment Prototypes Endowing Large La | — | — |
| `tri_subspaces_disentanglement_for_multim` | CVPR2026 | — | Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis | — | — |
| `lfd_rt**_cross_lingual_msa,_language_family_disentanglement` | Findings of ACL 2025 | — | LFD-RT** Cross-lingual MSA, Language Family Disentanglement | [链接](https://github.com/ShuoyuGuan/LFD-RT) | — |
| `dear` | Findings of ACL 2026 | — | DEAR | — | ✓ |
| `fast_retrieval_and_slow_reasoning**_explainable_msa` | Findings of ACL 2026 | — | Fast Retrieval and Slow Reasoning** Explainable MSA | — | — |
| `mulcot_rd**_resource_limited_joint_msa_reasoning（cot_蒸馏）` | Findings of ACL 2026 | — | MulCoT-RD** Resource-Limited Joint MSA Reasoning（CoT 蒸馏） | [链接](https://github.com/123sghn/MulCoT-RD) | — |
| `uncertainty_calibrated_elastic_alignment**,_missing_modalities` | Findings of ACL 2026 | — | Uncertainty-Calibrated Elastic Alignment**, Missing Modalities | — | — |
| `tf_mamba**_text_enhanced_fusion_mamba,_missing_modalities` | Findings of EMNLP 2025 | — | TF-Mamba** Text-enhanced Fusion Mamba, Missing Modalities | [链接](https://github.com/codemous/TF-Mamba) | — |
| `cmad` | ICCV 2025, pp.4626-4636 | — | CMAD | [链接](https://github.com/YetZzzzzz/CMAD) | ✓ |
| `cmad_correlation_aware_and_modalities_aw` | ICCV2025 | — | CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimod | — | — |
| `mma` | NAACL 2025 长文 | — | MMA | [链接](https://github.com/MMA4MSA/MMA) | — |
| `mer_clip` | WACV 2025, pp.6115-6124 | — | MER-CLIP | — | — |
| `multi_interactive_memory_network_for_asp` | aaai2019 | — | Multi-Interactive Memory Network for Aspect Based Multimodal Sentiment | — | — |
| `vistanet_visual_aspect_attention_network` | aaai2019 | — | VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Ana | — | — |
| `trimodal_attention_module_for_multimodal` | aaai2020 | — | Trimodal Attention Module for Multimodal Sentiment Analysis (Student A | — | — |
| `robust_msa_understanding_the_impact_of_m` | aaai2023 | — | Robust-MSA: Understanding the Impact of Modality Noise on Multimodal S | — | — |
| `a_unified_self_distillation_framework_fo` | aaai2024 | — | A Unified Self-Distillation Framework for Multimodal Sentiment Analysi | — | — |
| `a_multi_focus_driven_multi_branch_networ` | aaai2025 | — | A Multi-Focus-Driven Multi-Branch Network for Robust Multimodal Sentim | — | — |
| `bridging_the_gap_for_test_time_multimoda` | aaai2025 | — | Bridging the Gap for Test-Time Multimodal Sentiment Analysis | — | — |
| `dlf_disentangled_language_focused_multim` | aaai2025 | — | DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis | — | — |
| `enriching_multimodal_sentiment_analysis` | aaai2025 | — | Enriching Multimodal Sentiment Analysis Through Textual Emotional Desc | — | — |
| `msamba_exploring_multimodal_sentiment_an` | aaai2025 | — | MSAmba: Exploring Multimodal Sentiment Analysis with State Space Model | — | — |
| `mse_adapter_a_lightweight_plugin_endowin` | aaai2025 | — | MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to | — | — |
| `semi_iin_semi_supervised_intra_inter_mod` | aaai2025 | — | Semi-IIN: Semi-Supervised Intra-Inter Modal Interaction Learning Netwo | — | — |
| `towards_multimodal_sentiment_analysis_vi` | aaai2025 | — | Towards Multimodal Sentiment Analysis via Hierarchical Correlation Mod | — | — |
| `a_text_routed_sparse_mixture_of_experts` | aaai2026 | — | A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Tem | — | — |
| `fine_factorized_multimodal_sentiment_ana` | aaai2026 | — | FINE: Factorized Multimodal Sentiment Analysis via Mutual INformation  | — | — |
| `group_aware_multiscale_ensemble_learning` | aaai2026 | — | Group-aware Multiscale Ensemble Learning for Test-Time Multimodal Sent | — | — |
| `mdf_a_modality_aware_disentanglement_and` | aaai2026 | — | MDF: A Modality-Aware Disentanglement and Fusion Framework for Multimo | — | — |
| `pase_prototype_aligned_calibration_and_s` | aaai2026 | — | PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for  | — | — |
| `psa_mf_personality_sentiment_aligned_mul` | aaai2026 | — | PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimoda | — | — |
| `recovering_coherent_affective_patterns_a` | aaai2026 | — | Recovering Coherent Affective Patterns: Addressing Modality Missing in | — | — |
| `tmdc_a_two_stage_modality_denoising_and` | aaai2026 | — | TMDC: A Two-Stage Modality Denoising and Complementation Framework for | — | — |
| `multimodal_language_analysis_in_the_wild` | acl2018-1 | — | Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interp | — | — |
| `ch_sims_a_chinese_multimodal_sentiment_a` | acl2020 | — | CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-gra | — | — |
| `ctfn_hierarchical_learning_for_multimoda` | acl2021-1 | — | CTFN: Hierarchical Learning for Multimodal Sentiment Analysis Using Co | — | — |
| `multimodal_sentiment_detection_based_on` | acl2021-1 | — | Multimodal Sentiment Detection Based on Multi-channel Graph Neural Net | — | — |
| `a_text_centered_shared_private_framework` | acl2021f | — | A Text-Centered Shared-Private Framework via Cross-Modal Prediction fo | — | — |
| `msctd_a_multimodal_sentiment_chat_transl` | acl2022-1 | — | MSCTD: A Multimodal Sentiment Chat Translation Dataset | — | — |
| `m_sena_an_integrated_platform_for_multim` | acl2022-d | — | M-SENA: An Integrated Platform for Multimodal Sentiment Analysis | — | — |
| `sentiment_word_aware_multimodal_refineme` | acl2022f | — | Sentiment Word Aware Multimodal Refinement for Multimodal Sentiment An | — | — |
| `confede_contrastive_feature_decompositio` | acl2023-1 | contrastive | ConFEDE: Contrastive Feature Decomposition for Multimodal Sentiment An | — | — |
| `tackling_modality_heterogeneity_with_mul` | acl2023-1 | — | Tackling Modality Heterogeneity with Multi-View Calibration Network fo | — | — |
| `conki_contrastive_knowledge_injection_fo` | acl2023f | contrastive | ConKI: Contrastive Knowledge Injection for Multimodal Sentiment Analys | — | — |
| `sentiment_knowledge_enhanced_self_superv` | acl2023f | — | Sentiment Knowledge Enhanced Self-supervised Learning for Multimodal S | — | — |
| `proxy_driven_robust_multimodal_sentiment` | acl2025-1 | — | Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data | — | — |
| `cross_lingual_multimodal_sentiment_analy` | acl2025f | — | Cross-lingual Multimodal Sentiment Analysis for Low-Resource Languages | — | — |
| `beyond_static_alignment_adaptive_arbitra` | acl2026-1 | — | Beyond Static Alignment: Adaptive Arbitration for Semantic Incongruenc | — | — |
| `qa_moe_towards_a_continuous_reliability` | acl2026-1 | — | QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware M | — | — |
| `dear_distributional_error_aware_reliabil` | acl2026f | — | DEAR: Distributional Error-Aware Reliability for Robust Multimodal Sen | — | — |
| `fast_retrieval_and_slow_reasoning_for_ex` | acl2026f | — | Fast Retrieval and Slow Reasoning for Explainable Multimodal Sentiment | — | — |
| `resource_limited_joint_multimodal_sentim` | acl2026f | — | Resource-Limited Joint Multimodal Sentiment Reasoning and Classificati | — | — |
| `uncertainty_calibrated_elastic_alignment` | acl2026f | — | Uncertainty-Calibrated Elastic Alignment for Multimodal Sentiment Anal | — | — |
| `correlation_decoupled_knowledge_distilla` | cvpr2024 | — | Correlation-Decoupled Knowledge Distillation for Multimodal Sentiment  | — | — |
| `contextual_inter_modal_attention_for_mul` | emnlp2018 | — | Contextual Inter-modal Attention for Multi-modal Sentiment Analysis | — | — |
| `context_aware_interactive_attention_for` | emnlp2019-1 | — | Context-aware Interactive Attention for Multi-modal Sentiment and Emot | — | — |
| `which_is_making_the_contribution_modulat` | emnlp2021f | — | Which is Making the Contribution: Modulating Unimodal and Cross-modal  | — | — |
| `mitigating_inconsistencies_in_multimodal` | emnlp2022 | — | Mitigating Inconsistencies in Multimodal Sentiment Analysis under Unce | — | — |
| `unimse_towards_unified_multimodal_sentim` | emnlp2022 | — | UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Reco | — | — |
| `multimodal_contrastive_learning_via_uni` | emnlp2022f | contrastive | Multimodal Contrastive Learning via Uni-Modal Coding and Cross-Modal P | — | — |
| `descriptive_prompt_paraphrasing_for_targ` | emnlp2023f | — | Descriptive Prompt Paraphrasing for Target-Oriented Multimodal Sentime | — | — |
| `improving_multimodal_sentiment_analysis` | emnlp2023f | contrastive | Improving Multimodal Sentiment Analysis: Supervised Angular margin-bas | — | — |
| `rethinkingtmsc_an_empirical_study_for_ta` | emnlp2023f | — | RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sent | — | — |
| `visual_elements_mining_as_prompts_for_in` | emnlp2023f | — | Visual Elements Mining as Prompts for Instruction Learning for Target- | — | — |
| `d2r_dual_branch_dynamic_routing_network` | emnlp2024 | — | D2R: Dual-Branch Dynamic Routing Network for Multimodal Sentiment Dete | — | — |
| `knowledge_guided_dynamic_modality_attent` | emnlp2024f | — | Knowledge-Guided Dynamic Modality Attention Fusion Framework for Multi | — | — |
| `dual_path_dynamic_fusion_with_learnable` | emnlp2025 | — | Dual-Path Dynamic Fusion with Learnable Query for Multimodal Sentiment | — | — |
| `tf_mamba_text_enhanced_fusion_mamba_with` | emnlp2025f | — | TF-Mamba: Text-enhanced Fusion Mamba with Missing Modalities for Robus | [链接](https://github.com/codemous/TF-Mamba) | — |
| `adapting_bert_for_target_oriented_multim` | ijcai2019 | — | Adapting BERT for Target-Oriented Multimodal Sentiment Classification | — | — |
| `deepcu_integrating_both_common_and_uniqu` | ijcai2019 | — | DeepCU: Integrating both Common and Unique Latent Information for Mult | — | — |
| `targeted_multimodal_sentiment_classifica` | ijcai2022 | — | Targeted Multimodal Sentiment Classification based on Coarse-to-Fine G | — | — |
| `hydiscgan_a_hybrid_distributed_cgan_for` | ijcai2024 | — | HyDiscGAN: A Hybrid Distributed cGAN for Audio-Visual Privacy Preserva | — | — |
| `decoupling_and_reconstructing_a_multimod` | ijcai2025 | — | Decoupling and Reconstructing: A Multimodal Sentiment Analysis Framewo | — | — |
| `dfmu_distribution_based_framework_for_mo` | ijcai2025 | — | DFMU: Distribution-based Framework for Modeling Aleatoric Uncertainty  | — | — |
| `effective_sentiment_relevant_word_select` | mm2019 | — | Effective Sentiment-relevant Word Selection for Multi-modal Sentiment  | — | — |
| `summary_of_muse_2020_multimodal_sentimen` | mm2020 | — | Summary of MuSe 2020: Multimodal Sentiment Analysis, Emotion-target En | — | — |
| `transformer_based_feature_reconstruction` | mm2021 | — | Transformer-based Feature Reconstruction Network for Robust Multimodal | — | — |
| `counterfactual_reasoning_for_out_of_dist` | mm2022 | — | Counterfactual Reasoning for Out-of-distribution Multimodal Sentiment  | — | — |
| `cubemlp_an_mlp_based_model_for_multimoda` | mm2022 | — | CubeMLP: An MLP-based Model for Multimodal Sentiment Analysis and Depr | — | — |
| `acformer_an_aligned_and_compact_transfor` | mm2023 | — | AcFormer: An Aligned and Compact Transformer for Multimodal Sentiment  | — | — |
| `building_robust_multimodal_sentiment_rec` | mm2023 | — | Building Robust Multimodal Sentiment Recognition via a Simple yet Effe | — | — |
| `cross_modality_representation_interactiv` | mm2023 | — | Cross-modality Representation Interactive Learning for Multimodal Sent | — | — |
| `few_shot_multimodal_sentiment_analysis_b` | mm2023 | — | Few-shot Multimodal Sentiment Analysis Based on Multimodal Probabilist | — | — |
| `general_debiasing_for_multimodal_sentime` | mm2023 | — | General Debiasing for Multimodal Sentiment Analysis | — | — |
| `enhanced_experts_with_uncertainty_aware` | mm2024 | — | Enhanced Experts with Uncertainty-Aware Routing for Multimodal Sentime | — | — |
| `glomo_global_local_modal_fusion_for_mult` | mm2024 | — | GLoMo: Global-Local Modal Fusion for Multimodal Sentiment Analysis | — | — |
| `grace_gradient_based_active_learning_wit` | mm2024 | — | GRACE: GRadient-based Active Learning with Curriculum Enhancement for  | — | — |
| `kebr_knowledge_enhanced_self_supervised` | mm2024 | — | KEBR: Knowledge Enhanced Self-Supervised Balanced Representation for M | — | — |
| `learning_in_order_a_sequential_strategy` | mm2024 | — | Learning in Order! A Sequential Strategy to Learn Invariant Features f | — | — |
| `robust_multimodal_sentiment_analysis_of` | mm2024 | — | Robust Multimodal Sentiment Analysis of Image-Text Pairs by Distributi | — | — |
| `wisdom_improving_multimodal_sentiment_an` | mm2024 | — | WisdoM: Improving Multimodal Sentiment Analysis by Fusing Contextual W | — | — |
| `ddse_a_decoupled_dual_stream_enhanced_fr` | mm2025 | — | DDSE: A Decoupled Dual-Stream Enhanced Framework for Multimodal Sentim | — | — |
| `diffufuse_diffusion_driven_dual_stream_f` | mm2025 | — | DiffuFuse: Diffusion-Driven Dual-Stream Fusion Framework for Multimoda | — | — |
| `impact_of_stickers_on_multimodal_sentime` | mm2025 | — | Impact of Stickers on Multimodal Sentiment and Intent in Social Media: | — | — |
| `ldw_label_divergence_weighting_for_multi` | mm2025 | — | LDW: Label Divergence Weighting for Multimodal Sentiment Analysis | — | — |
| `towards_explainable_fusion_and_balanced` | mm2025 | — | Towards Explainable Fusion and Balanced Learning in Multimodal Sentime | — | — |
| `analyzing_modality_robustness_in_multimo` | naacl2022 | — | Analyzing Modality Robustness in Multimodal Sentiment Analysis | — | — |

## ✕ 已排除（6）

| key | 出处 | 类别 | 标题 | 作者代码 | PDF |
|---|---|---|---|---|---|
| `p_rmf` | ACL 2025 长文 | — | P-RMF | [链接](https://github.com/aoqzhu/P-RMF) | — |
| `ebmc` | CVPR 2026, pp.30183-30193 | different-features | EBMC | [链接](https://github.com/kangverse/EBMC) | ✓ |
| `enhance_then_balance_modality_collaborat` | CVPR2026 | different-features | Enhance-then-Balance Modality Collaboration for Robust Multimodal Sent | [链接](https://github.com/kangverse/EBMC) | — |
| `molan` | Findings of ACL 2026 | — | MoLAN | [链接](https://github.com/betterfly123/MoLAN-Framework) | ✓ |
| `molan_a_unified_modality_aware_noise_dyn` | acl2026f | — | MoLAN: A Unified Modality-Aware Noise Dynamic Editing Framework for Mu | [链接](https://github.com/betterfly123/MoLAN-Framework) | — |
| `clmlf_a_contrastive_learning_and_multi_l` | naacl2022f | contrastive | CLMLF: A Contrastive Learning and Multi-Layer Fusion Method for Multim | — | — |
