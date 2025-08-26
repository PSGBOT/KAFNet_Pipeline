# PLSGB Post SSAM Pipeline

## Process

1. VLM Inference:

    - Input: SSAM output (level seg dataset + part seg dataset)
    - Output: VLM output + instance description
    - Source Folder: pixtral-12B-inference
    - Code: vlm.py

2. KAF (FCSGG) Generation:

    - Input: SSAM output (level seg dataset + part seg dataset)
    - Output: KAF out
    - Source Folder: KAF-Generation
    - Code: FCSGG_api.py

3. Matching:

    - Input: SSAM output (level seg dataset + part seg dataset) + VLM output + instance description + KAF out
    - Output: Scene graph
    - Code: constructor.py

4. Task Planning:

    - Input: Scene graph
    - Output: task
    - Source Folder: Sayplan_Reconstruct
    - Code: None

## Overall Pipeline File

test_pipeline.py
