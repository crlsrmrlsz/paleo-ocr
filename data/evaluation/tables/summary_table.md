# Summary Metrics Table

Combined results across all 92 ground truth comparisons (37 CODEA paleographic + 37 CODEA critical + 18 Toledo editorial). Models ranked by CER. See the main [README](../../../README.md) for the curated 55-page subset and [full metrics](../full_metrics.md) for per-tier Romein breakdowns.

| Rank | Model | Pages | CER | CER_n | NLS | WER | Precision | Recall | BOC |
|------|-------|-------|-----|-------|-----|-----|-----------|--------|-----|
| 1 | baseline/transkribus_spanish_sage | 92 | 0.3766 | 0.3316 | 0.6460 | 0.8779 | 0.7243 | 0.7480 | 0.2071 |
| 2 | pipeline/11_yolo_crop_anti_halluc | 92 | 0.3803 | 0.3472 | 0.6301 | 0.7833 | 0.7229 | 0.7033 | 0.1974 |
| 3 | pipeline/10_yolo_crop_bare_merge | 92 | 0.3813 | 0.3467 | 0.6317 | 0.7880 | 0.7185 | 0.7034 | 0.1934 |
| 4 | pipeline/09_yolo_crop_anti_delete | 92 | 0.3827 | 0.3476 | 0.6303 | 0.7915 | 0.7168 | 0.7036 | 0.1945 |
| 5 | pipeline/07_yolo_crop_strips | 92 | 0.3856 | 0.3504 | 0.6276 | 0.7908 | 0.7195 | 0.7003 | 0.2001 |
| 6 | pipeline/04b_strips_split_ctx_rezoom | 92 | 0.3890 | 0.3513 | 0.6265 | 0.8002 | 0.7130 | 0.7047 | 0.2044 |
| 7 | pipeline/16_denoise | 92 | 0.3895 | 0.3549 | 0.6217 | 0.7964 | 0.7111 | 0.6971 | 0.2018 |
| 8 | pipeline/03_strips_page_desc | 92 | 0.3924 | 0.3517 | 0.6257 | 0.8024 | 0.7074 | 0.7058 | 0.2000 |
| 9 | pipeline/06_strips_no_context | 92 | 0.3926 | 0.3528 | 0.6240 | 0.7936 | 0.7085 | 0.7044 | 0.1985 |
| 10 | pipeline/08_yolo_crop_noise_warn | 92 | 0.3942 | 0.3711 | 0.6108 | 0.7629 | 0.7533 | 0.6675 | 0.2381 |
| 11 | pipeline/04_strips_split_context | 92 | 0.3942 | 0.3553 | 0.6225 | 0.8109 | 0.7069 | 0.7022 | 0.2046 |
| 12 | pipeline/14_clahe | 92 | 0.3981 | 0.3595 | 0.6173 | 0.8132 | 0.7060 | 0.6943 | 0.2080 |
| 13 | pipeline/15_sharpen | 92 | 0.4007 | 0.3619 | 0.6152 | 0.8142 | 0.7034 | 0.6941 | 0.2038 |
| 14 | pipeline/02_yolo_model_first | 92 | 0.4046 | 0.3757 | 0.6032 | 0.7852 | 0.7293 | 0.6677 | 0.2301 |
| 15 | baseline/claude_opus_4_6 | 92 | 0.4293 | 0.3917 | 0.5848 | 0.8533 | 0.6970 | 0.6627 | 0.2509 |
| 16 | pipeline/01_yolo_blocks | 92 | 0.4328 | 0.3939 | 0.5866 | 0.8224 | 0.7109 | 0.6643 | 0.2663 |
| 17 | pipeline/13_yolo_crop_anti_halluc_gemini31 | 82 | 0.4640 | 0.4473 | 0.5405 | 0.8391 | 0.7717 | 0.5870 | 0.3738 |
| 18 | baseline/tridis_v2 | 92 | 0.4963 | 0.4729 | 0.5101 | 0.8077 | 0.7384 | 0.5693 | 0.3610 |
| 19 | pipeline/05_tiles_2d | 92 | 0.5040 | 0.4409 | 0.5267 | 0.8906 | 0.6308 | 0.6246 | 0.2239 |
| 20 | pipeline/12_yolo_crop_anti_halluc_gpt54 | 90 | 0.5573 | 0.5457 | 0.4428 | 0.8183 | 0.7335 | 0.4690 | 0.3900 |
| 21 | baseline/google_cloud_vision | 92 | 0.6038 | 0.5844 | 0.3975 | 0.9461 | 0.6310 | 0.4347 | 0.4495 |
| 22 | baseline/gpt_5_4 | 92 | 0.6226 | 0.6123 | 0.3777 | 0.8690 | 0.5756 | 0.3999 | 0.4768 |
| 23 | baseline/mistral_large_3 | 92 | 0.7294 | 0.6365 | 0.3311 | 1.0481 | 0.4562 | 0.4163 | 0.3788 |
| 24 | baseline/gemini_3_1_pro | 92 | 0.7693 | 0.7629 | 0.2307 | 0.9364 | 0.6560 | 0.2419 | 0.7237 |
| 25 | baseline/qwen3_vl_8b | 92 | 1.4021 | 0.6715 | 0.3163 | 2.1165 | 0.4761 | 0.4744 | 0.6395 |
| 26 | baseline/churro_3b | 92 | 2.4482 | 0.5037 | 0.4540 | 2.8878 | 0.5497 | 0.7070 | 0.3184 |
