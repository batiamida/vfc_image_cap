# Workflow
1. Load images
   - Best-case scenario: UAV images with aircrafts, soldiers, buildings, different equipments, other military machinery on them
   - Average: UAV + other areal images mix with some military machinery on it and maybe soldiers
   - Worst: mix of non-aerial and areal images with mix of military and non-military images
2. Find open-source models to build our VisualFactChecker pipeline & make the best performing model as base-model
3. Make data (images) format acceptable by the base-model and our pipeline
4. Build VisualFactChecker pipeline with chosen models
5. Generate some annotations with base-model and our pipeline
6. Evaluation
   - Compare base-model and pipeline which one has less hallucinations
   - Compare differencec in captions generation time
