# CoVA: Text-guided Composed Video Retrieval For Audio-Visual Content
Impleement of paper: [CoVA: Text-guided Composed Video Retrieval For Audio-Visual Content](https://perceptualai-lab.github.io/CoVA/)

## 🔍 Project Page

Visit our project page for additional information and interactive examples:
* **[CoVA: Text-guided Composed Video Retrieval For Audio-Visual Content](https://perceptualai-lab.github.io/CoVA/)**

## 🐳 Docker

Our implementation is also available as a Docker image:
* [Docker Hub](https://hub.docker.com/repository/docker/qkenr0804/goal/general)

```bash
# Pull the image
docker push jshfu/cova:latest
```

## 🏋️ Pre-trained Weights

Download CLIP (ViT-B/32) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

Download CLIP (ViT-B/16) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

Download AST weight from [AST](https://github.com/YuanGongND/ast) (Pretrained Models 1:"Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)").

## 📊 Datasets

Please download the datasets from the links below:

* AudioCaps Dataset
    * [Download link](https://audiocaps.github.io/)

## 🚀 Training

```bash
sh run.sh
```