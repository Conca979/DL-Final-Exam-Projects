#!/usr/bin/env bash
set -euo pipefail

# Declare project folder structures, names, and requirements
projects=(
  "01_label_efficient_classification|Label-Efficient Image Classification with Vision Foundation Models|Investigate how pretrained vision foundation models perform when only a limited amount of labeled data is available.%Compare conventional transfer learning with a pretrained vision foundation model using different amounts of labeled training data."
  "02_fine_grained_recognition|Fine-Grained Image Recognition under Limited Training Data|Investigate the classification of visually similar categories when only a limited amount of labeled training data is available.%Analyze the effects of transfer learning, data augmentation, and training-data size on classification performance."
  "03_satellite_domain_classification|Satellite Image Classification across Different Domains|Investigate how image-classification models perform when the training and test images come from different regions, conditions, or data sources.%Compare conventional CNN-based models with pretrained vision models under these changes in data distribution."
  "04_industrial_anomaly_detection|Industrial Visual Anomaly Detection with Pretrained Features|Detect abnormal or defective products when most training images contain only normal samples.%Compare conventional anomaly-detection methods with approaches based on pretrained vision models."
  "05_robust_image_degradation|Robust Image Recognition under Realistic Image Degradations|Investigate how image-classification models perform when input images are affected by blur, noise, low light, or other common degradations.%Compare the robustness of different deep-learning models under changing image quality."
  "06_open_vocab_detection|Open-Vocabulary Object Detection with Text Prompts|Detect objects using text descriptions instead of relying only on a fixed set of predefined classes.%Investigate how different object names, synonyms, and descriptive prompts affect detection performance."
  "07_interactive_segmentation_prompts|Effects of User Prompts on Interactive Image Segmentation|Investigate how different types and qualities of user prompts affect image-segmentation performance.%Compare point prompts, box prompts, and inaccurate prompts, and analyze common failure cases."
  "08_cross_domain_vision_models|Applying Pretrained Vision Models across Different Image Domains|Investigate how well pretrained vision models can be applied to different types of image data.%Compare model performance when the training and test images come from different domains, such as natural, satellite, medical, or industrial images."
  "09_vietnamese_image_captioning|Vietnamese Image Captioning with Vision-Language Models|Generate Vietnamese descriptions for images using lightweight vision-language models.%Evaluate caption quality and investigate how prompt design affects the generated descriptions and common errors."
  "10_vqa_question_types|Visual Question Answering across Different Question Types|Develop a system that answers text-based questions about image content.%Evaluate performance across different question types, including object recognition, color, counting, spatial relationships, and visual reasoning."
  "11_vqa_receipts_documents|Visual Question Answering for Receipts and Documents|Extract and understand information from receipt and document images using deep-learning models.%Compare traditional OCR-based methods with modern vision-language approaches for answering questions about document content."
  "12_chart_understanding_vlm|Chart Understanding with Vision-Language Models|Investigate how vision-language models understand and answer questions about bar charts, line charts, and other common charts.%Evaluate performance on factual questions, numerical questions, and visual reasoning, and analyze common failure cases."
  "13_multimodal_meme_understanding|Multimodal Meme Understanding: Image, Text, or Both?|Investigate how visual and textual information contribute to understanding the meaning of internet memes.%Compare image-only, text-only, and multimodal models, and analyze when combining both modalities improves performance."
  "14_multimodal_food_understanding|Multimodal Food Understanding from Images and Text|Recognize food categories and extract useful information from both food images and related text.%Compare image-only, text-only, and multimodal approaches to determine whether combining both modalities improves food recognition and understanding."
  "15_video_qa_multimodal|Video Question Answering with Multimodal Models|Answer text-based questions about short video clips using lightweight pretrained multimodal models.%Investigate how frame selection, temporal information, and different question types affect answering performance."
  "16_image_inpainting_masks|Deep Image Inpainting under Different Missing-Region Patterns|Restore missing or damaged regions in images using deep-learning models.%Compare restoration performance across different mask shapes, mask sizes, and missing-region patterns."
  "17_super_resolution_unknown_degradation|Image Super-Resolution under Unknown Degradations|Recover high-resolution images from degraded low-resolution inputs using deep-learning models.%Evaluate how super-resolution models perform when test images contain blur, noise, or compression conditions that were not included during training."
  "18_low_light_enhancement|Low-Light Image Enhancement and Downstream Recognition|Enhance images captured under low-light conditions using deep-learning models.%Investigate whether improved visual quality also leads to better performance on downstream recognition tasks."
  "19_synthetic_images_imbalance|Can Synthetic Images Improve Imbalanced Classification?|Generate synthetic images for minority classes to reduce class imbalance in the training data.%Compare synthetic-data augmentation with conventional data augmentation and oversampling, and evaluate their effects on classification performance."
  "20_text_to_image_lora_personalization|Personalizing Text-to-Image Models with Lightweight Fine-Tuning|Adapt a pretrained text-to-image model to generate images of a new visual concept using only a small training set.%Investigate how the amount of training data and different LoRA settings affect the quality and consistency of generated images."
  "21_old_photo_restoration|Deep Restoration of Old and Damaged Photographs|Restore old photographs affected by noise, scratches, missing regions, or other visual degradations using deep-learning models.%Investigate how different restoration steps or restoration pipelines influence the final visual quality."
  "22_ood_detection|Can a Neural Network Know When It Does Not Know?|Investigate whether a neural network can recognize inputs that are different from the data used during training.%Compare different methods for detecting unfamiliar or out-of-distribution inputs based on model confidence and learned features."
  "23_confidence_calibration|When Can We Trust Neural Network Confidence?|Study whether prediction confidence accurately reflects the probability of being correct.%Evaluate calibration techniques under normal conditions and distribution shifts."
  "24_knowledge_distillation|Knowledge Distillation: How Small Can a Neural Network Become?|Train a smaller student model to learn from a larger and more powerful teacher model.%Investigate the trade-off between model size, computational efficiency, and prediction performance."
  "25_model_compression_efficiency|Efficient Deep Learning through Quantization, Pruning, and Distillation|Apply quantization, pruning, and knowledge distillation to reduce the memory and computational requirements of deep neural networks.%Compare these techniques and analyze the trade-off between model efficiency and prediction performance."
  "26_continual_learning_classification|Continual Image Classification without Catastrophic Forgetting|Train an image-classification model to learn new classes or tasks over time while preserving previously learned knowledge.%Compare simple sequential training with methods designed to reduce forgetting of earlier classes or tasks."
  "27_federated_learning_noniid|Federated Image Classification under Non-IID Data|Train an image-classification model collaboratively across multiple simulated clients without directly sharing their local training data.%Investigate how differences in data distributions across clients affect federated-learning performance."
  "28_chest_xray_multilabel|Multi-Label Chest X-Ray Classification with Confidence Calibration|Develop a deep-learning model that can identify multiple findings that may appear simultaneously in a chest X-ray image.%Investigate the effects of class imbalance, prediction thresholds, and confidence reliability on classification performance."
  "29_skin_lesion_imbalance|Skin-Lesion Classification under Severe Class Imbalance|Develop an image-classification model for skin lesions when some categories contain far fewer training samples than others.%Compare different sampling, loss-function, and data-augmentation strategies for handling class imbalance."
  "30_explainable_retinal_classification|Explainable Retinal Image Classification|Classify retinal images using deep-learning models and visualize the image regions that contribute most to model predictions.%Compare different explanation methods and analyze whether the highlighted regions are consistent for correct and incorrect predictions."
  "31_medical_segmentation_limited_data|Medical Image Segmentation with Limited Training Data|Develop deep-learning models for segmenting anatomical structures or abnormalities when only a limited amount of annotated medical images is available.%Compare conventional segmentation models with pretrained models and investigate how training-data size affects segmentation performance."
  "32_histopathology_staining_robustness|Robust Histopathology Image Classification under Staining Variations|Classify histopathology images when differences in staining and image appearance are present.%Investigate how color normalization, data augmentation, and pretrained vision models affect classification robustness."
  "33_ecg_arrhythmia_classification|ECG Arrhythmia Classification with Deep Sequence Models|Classify different types of heart rhythms from ECG signals using deep-learning models.%Compare convolutional, recurrent, and Transformer-based approaches and analyze their performance under different evaluation settings."
  "34_speech_emotion_recognition|Robust Speech Emotion Recognition with Pretrained Audio Models|Recognize emotions from speech using conventional audio features and pretrained deep-learning audio models.%Evaluate how noise and differences between speakers affect emotion-recognition performance."
  "35_tiny_keyword_spotting|Tiny Keyword Spotting under Noise and Resource Constraints|Recognize short spoken commands using compact deep-learning models suitable for resource-limited devices.%Investigate the trade-off between recognition accuracy, noise robustness, model size, and inference speed."
  "36_environmental_sound_recognition|Environmental Sound Recognition under Unseen Acoustic Conditions|Classify environmental sounds such as sirens, engines, animal sounds, and machinery using deep-learning models.%Investigate how well the models perform when test audio contains noise, reverberation, or acoustic conditions not encountered during training."
  "37_human_activity_recognition|Human Activity Recognition with Multi-Sensor Deep Learning|Recognize human activities from wearable-sensor signals, such as accelerometer and gyroscope data, using deep-learning models.%Compare different sensor combinations and deep sequence models to determine how multiple sensors affect recognition performance."
  "38_time_series_anomaly_detection|Deep Time-Series Anomaly Detection under Changing Conditions|Detect unusual or abnormal patterns in time-series data using deep-learning models.%Investigate how noise, changes in operating conditions, and different detection thresholds affect anomaly-detection performance."
  "39_few_shot_video_action_recognition|Few-Shot Video Action Recognition with Pretrained Video Models|Recognize human actions in short video clips when only a limited number of labeled training examples are available.%Investigate how pretrained video models, frame-sampling strategies, and training-data size affect action-recognition performance."
)

for item in "${projects[@]}"; do
  IFS='|' read -r folder name reqs <<< "$item"
  mkdir -p "$folder"

  # Format requirements list
  formatted_reqs=""
  IFS='%' read -ra req_arr <<< "$reqs"
  for req in "${req_arr[@]}"; do
    formatted_reqs+="- ${req}"$'\n'
  done

  # Write README.md
  cat <<EOF > "$folder/README.md"
# $name

## Project Requirements
$formatted_reqs
EOF
done

echo "Created all 39 project folders with README.md files successfully!"
