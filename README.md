# XAI-Banking-Compliance

# Neural Network Interpretation for Regulatory Compliance in Banking: Explainable AI (XAI) for Banca Virtuosa Project

This notebook demonstrates the application of Explainable AI (XAI) techniques to a neural network model, specifically a DenseNet121 fine-tuned on the MNIST dataset. The project aims to address regulatory compliance requirements for high-risk AI systems in financial services, aligning with EBA/GL/2020/06 guidelines and the EU AI Act transparency requirements.

## Project Overview

The project is structured into three phases:

*   **Phase 1:** Fine-tuning of a pre-trained DenseNet121 model on the MNIST dataset.
*   **Phase 2:** Generation of saliency maps using five different XAI techniques.
*   **Phase 3:** A final report detailing correct vs. incorrect prediction analysis and an explainable baseline system.

## Regulatory Context

This work directly addresses the need for AI transparency and explainability, crucial for regulatory compliance in the banking sector. It aligns with:

*   **EBA/GL/2020/06:** Guidelines on internal governance, emphasizing robust AI systems.
*   **EU AI Act:** Transparency requirements for high-risk AI, ensuring clarity in decision-making processes.

## Dataset: MNIST

The MNIST (Modified National Institute of Standards and Technology) dataset, comprising handwritten digits (0–9), serves as a proxy for document-level pattern recognition in financial OCR pipelines. The dataset characteristics are:

*   **Samples:** 60,000 (train), 10,000 (test)
*   **Image size:** 28 × 28 px (grayscale), resized to 64 × 64 for DenseNet compatibility.
*   **Channels:** Grayscale images are replicated to 3 channels to match ImageNet pre-trained model input.
*   **Classes:** 10 (digits 0–9), with a balanced distribution.

## Model Architecture: DenseNet121 (Transfer Learning)

A DenseNet121 model, pre-trained on ImageNet, is used as the backbone. The model leverages dense skip connections for improved gradient flow and parameter efficiency. The strategy involves:

*   **Head-only fine-tuning:** Freezing the convolutional weights and replacing the final fully-connected layer to adapt to 10 classes.
*   **Custom head:** `Linear(1024) → ReLU → Linear(256) → ReLU → Linear(10)`.
*   **Achieved Accuracy:** Approximately 94.39% on the test set.

## Explainable AI (XAI) Techniques

Five distinct XAI methods are applied and compared:

1.  **Grad-CAM (Gradient-weighted Class Activation Mapping):** Identifies which regions of an image are important for the model's decision. (Region-level, gradient-based)
2.  **Integrated Gradients:** Provides pixel-level attribution by integrating gradients along a path from a baseline to the input image. (Pixel-level, gradient-based, axiomatic guarantees)
3.  **LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions by perturbing the input and training a local sparse linear model. (Segment-level, model-agnostic)
4.  **SHAP (SHapley Additive exPlanations):** Assigns a Shapley value to each feature (pixel) indicating its contribution to the prediction. (Pixel-level, game-theoretic, axiomatic guarantees)
5.  **Occlusion Maps:** Systematically occludes parts of the input image to measure the impact on prediction probability. (Region-level, model-agnostic, black-box compatible)

## Key Findings

*   **Correct Predictions:** XAI methods show concentrated attribution on the discriminative stroke structures of digits, indicating robust and well-justified decisions. High confidence and coherent activation maps are observed.
*   **Incorrect Predictions:** XAI methods reveal diffuse or fragmented attributions, often highlighting ambiguous regions or areas shared with confusable digits. Low confidence and scattered activation maps are characteristic.
*   **Attribution Concentration:** A quantitative metric (pixel fraction for 80% attribution mass) confirms that correct predictions have more focused attributions, while incorrect ones exhibit broader, more diffuse explanations. Occlusion Maps show the most diagnostic gap between correct and incorrect predictions.
*   **Consistent Failure Modes:** All five XAI methods consistently identify the same types of misclassifications (e.g., 0↔6, 3↔6, 7↔2, 9↔4), confirming genuine model weaknesses rather than technique artifacts.

## Explainable Baseline System

An interpretable Decision Tree classifier, trained on PCA-reduced DenseNet features, serves as an explainable baseline. While achieving 54.09% accuracy (compared to DenseNet's 94.39%), it illustrates the **accuracy-interpretability trade-off**. This approach justifies the use of post-hoc XAI for high-accuracy black-box models, recovering interpretability without sacrificing performance.

## Regulatory Compliance Assessment

The project comprehensively meets various regulatory requirements:

*   **Transparency & Explainability:** Achieved through the application of multiple XAI techniques and a comparative analysis.
*   **Auditability:** Attributions are computed from model weights, allowing for verification.
*   **Model Validation & Error Analysis:** Demonstrated through confusion matrices, per-class accuracy, and detailed correct vs. incorrect saliency comparisons.
*   **Interpretable Alternative:** Provided by the Decision Tree baseline with feature importance.

## Recommendations for Production Deployment

1.  **SHAP:** Utilize as the primary method for production reports due to its axiomatic guarantees.
2.  **Occlusion Maps:** Employ for regulator-facing documentation due to its intuitive nature.
3.  **Grad-CAM:** Implement in live monitoring dashboards for fast inference-time insights.
4.  **Re-train DenseNet:** With domain-specific augmentations and fine-tuning for improved performance.
5.  **Log Attribution Maps:** Maintain audit trails by logging all attribution maps alongside predictions.
6.  **Confidence Thresholds:** Flag predictions below 70% confidence for human review.
7.  **Quarterly Drift Monitoring:** Conduct regular monitoring of attribution distributions to detect dataset shifts.
