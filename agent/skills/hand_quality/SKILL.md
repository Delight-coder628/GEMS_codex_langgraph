# Skill: Hand Quality Verification and Prompt Refinement

## Description
Improve visible human hands, finger anatomy, grasping poses, occlusion, and hand-object interactions. Trigger this skill when hands, fingers, holding, touching, or grasping are important to the requested image.

## Instructions
When visible hands or fingers matter:

1. State the required number and placement of visible hands.
2. Require anatomically plausible hands and natural wrist alignment.
3. Require five distinct fingers when all fingers should be visible.
4. Avoid duplicated, fused, melted, missing, or extra fingers.
5. Describe natural contact between each hand and any held object.
6. Preserve realistic occlusion when fingers overlap an object or another hand.
7. During verification, inspect anatomy, finger count, finger separation, pose, and object contact.
8. Tag failures as `hand_error` and give a concrete correction for the next prompt.
