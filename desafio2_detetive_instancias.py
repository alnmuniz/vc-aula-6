import os
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from ultralytics import YOLO

def download_sample_image(filename="baloes_ou_frutas.jpg"):
    """
    Baixa uma imagem de teste contendo múltiplas instâncias da mesma classe.
    """
    if os.path.exists(filename):
        os.remove(filename)
    
    # URL de imagem de exemplo do repositório ultralytics com múltiplos pedestres/objetos
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    print(f"Baixando imagem real de teste ({url})...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
        out_file.write(response.read())
    print("Download concluído com sucesso!")
    return filename

def run_mask_rcnn(image_path, score_threshold=0.5):
    """
    Executa a Mask R-CNN (torchvision) na imagem especificada.
    Retorna a imagem com as máscaras e caixas desenhadas.
    """
    print("\n=== Executando Mask R-CNN (ResNet-50-FPN) ===")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    
    categories = weights.meta["categories"]
    
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(pil_image)
    
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy() # (N, 1, H, W)
    
    # Filtrar por limiar de confiança
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]
    
    img_np = np.array(pil_image).copy()
    overlay = img_np.copy()
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
    
    print(f"Mask R-CNN detectou {len(boxes)} instâncias acima do limiar de {score_threshold}:")
    
    for idx in range(len(boxes)):
        box = boxes[idx].astype(int)
        label_name = categories[labels[idx]] if labels[idx] < len(categories) else str(labels[idx])
        score = scores[idx]
        mask = masks[idx, 0] > 0.5
        color = colors[idx].tolist()
        
        # Aplicar cor da máscara
        overlay[mask] = (overlay[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        
        # Desenhar caixa delimitadora e rótulo
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
        caption = f"M-RCNN #{idx+1} {label_name}: {score:.2f}"
        cv2.putText(overlay, caption, (box[0], max(box[1] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        print(f"  - Instância #{idx+1}: Classe '{label_name}', Confiança: {score:.2f}, Caixa: {box.tolist()}")
        
    return overlay, len(boxes)

def run_yolov8_seg(image_path, score_threshold=0.5):
    """
    Executa o YOLOv8-Seg (ultralytics) na imagem especificada.
    Retorna a imagem com as máscaras e caixas desenhadas.
    """
    print("\n=== Executando YOLOv8-Seg (yolov8n-seg.pt) ===")
    model = YOLO("yolov8n-seg.pt")
    results = model(image_path, conf=score_threshold)[0]
    
    plotted_bgr = results.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    
    num_instances = len(results.boxes) if results.boxes is not None else 0
    print(f"YOLOv8-Seg detectou {num_instances} instâncias com confiança >= {score_threshold}.")
    
    return plotted_rgb, num_instances

def main():
    img_filename = download_sample_image()
    
    mask_rcnn_img, count_mrcnn = run_mask_rcnn(img_filename)
    yolo_img, count_yolo = run_yolov8_seg(img_filename)
    
    orig_img = Image.open(img_filename).convert("RGB")
    
    # Plotar comparação visual lado a lado
    plt.figure(figsize=(21, 7))
    
    plt.subplot(1, 3, 1)
    plt.title("Imagem Original", fontsize=14, fontweight='bold')
    plt.imshow(orig_img)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title(f"Mask R-CNN ({count_mrcnn} instâncias)", fontsize=14, fontweight='bold')
    plt.imshow(mask_rcnn_img)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title(f"YOLOv8-Seg ({count_yolo} instâncias)", fontsize=14, fontweight='bold')
    plt.imshow(yolo_img)
    plt.axis("off")
    
    plt.tight_layout()
    output_path = "mask_rcnn_vs_yolov8.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"\nComparação salva com sucesso em '{output_path}'.")

if __name__ == "__main__":
    main()
