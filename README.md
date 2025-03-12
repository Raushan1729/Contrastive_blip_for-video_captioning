# Video Captioning using Contrastive BLIP approach with multiple NLP Models

## Overview
This project extracts frames from a video, generates captions for each frame using the BLIP model, clusters similar captions using cosine similarity, removes redundancy, and finally summarizes the captions using a transformer-based summarization model.

## Features
- Extracts frames from a video at a specified frame rate.
- Generates captions for each frame using the **Salesforce BLIP model**.
- Computes **cosine similarity** between captions to identify similarities.
- Groups similar captions to avoid redundancy.
- Cleans and refines captions to remove noise.
- Summarizes grouped captions using **BART-Large CNN**.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install opencv-python transformers sentence-transformers torch pillow scikit-learn
```

## Installation & Setup
1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repo](https://github.com/Raushan1729/Contrastive_blip_for-video_captioning.git
cd your-repo
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Ensure you have a GPU for optimal performance (optional).**

## Usage
### 1. Run Video Processing
Modify the `video_path` and `frame_output_folder` in `process_video()` and execute:
```python
video_path = 'path/to/video.mp4'
frame_output_folder = 'path/to/output/folder'
final_captions = process_video(video_path, frame_output_folder, frame_rate=1, similarity_threshold=0.8)

print("Final Video Captions:")
for i, caption in enumerate(final_captions, 1):
    print(f"Caption {i}: {caption}")
```
### 2. Explanation of Functions
- **`extract_frames_from_video()`**: Extracts frames from a video at a specified frame rate.
- **`generate_caption()`**: Uses BLIP to generate captions for each frame.
- **`compute_similarity()`**: Computes the similarity between captions.
- **`group_captions()`**: Groups captions based on a similarity threshold.
- **`remove_redundant_captions()`**: Removes duplicate or similar captions.
- **`clean_caption()`**: Cleans and refines caption text.
- **`summarize_groups()`**: Uses Hugging Face’s `facebook/bart-large-cnn` to generate concise summaries.

## Example Output
```
Final Video Captions:
Caption 1: Police officers are investigating the crime scene.
Caption 2: A suspect is being taken into custody by law enforcement.
Caption 3: A witness is giving a statement to the authorities.
```

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenCV](https://opencv.org/)

## License

This project is licensed under the MIT License.

