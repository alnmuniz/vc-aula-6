# Relatório Técnico: Desafio 2 - O Detetive de Instâncias (Mask R-CNN vs YOLOv8-Seg)

---

## 1. Visão Geral do Desafio
O objetivo deste desafio é explorar o paradigma da **Segmentação de Instâncias**, configurando o repositório clássico da **Mask R-CNN** (Matterport / He et al., 2017) e comparando seus resultados com a abordagem moderna de estágio único do **YOLOv8-Seg**.

---

## 2. Resultados Experimentais

A inferência foi realizada em uma imagem contendo múltiplas instâncias da classe `person` (pedestres sobrepostos em primeiro plano) e um `bus` ao fundo.

| Modelo | Instâncias Detectadas | Confiança Média | Tempo de Inferência |
| :--- | :--- | :--- | :--- |
| **Mask R-CNN (ResNet-50-FPN)** | **5 instâncias** (4 pessoas, 1 ônibus) | ~0.91 | ~500ms (CPU) |
| **YOLOv8-Seg (yolov8n-seg)** | **4 instâncias** (3 pessoas, 1 ônibus) | ~0.855 | ~190ms (CPU) |

![Comparação Visual Side-by-Side](mask_rcnn_vs_yolov8.png)

---

## 3. Análise Teórica e Resposta às Questões

### Questão 1: Qual é a diferença fundamental na visualização?
- **Detecção de Objetos Tradicional**: Retorna apenas caixas delimitadoras (*bounding boxes* retangulares), sem fornecer informações sobre a geometria exata dos contornos dos objetos.
- **Segmentação Semântica Pura**: Classifica cada pixel da imagem em uma categoria (ex: "tudo que é pessoa fica vermelho"). Se houver três pessoas sobrepostas, a saída será uma única mancha contínua de pixels vermelhos, sem contagem individual.
- **Segmentação de Instâncias (Mask R-CNN e YOLOv8-Seg)**: Atribui um **rótulo de instância único e uma máscara binária pixel a pixel para cada objeto individual**. Visualmente, cada pessoa/objeto recebe uma **cor distinta** (ex: Pessoa 1 = verde, Pessoa 2 = azul, Pessoa 3 = amarelo), permitindo separar, contar e isolar objetos mesmo que pertençam à mesma classe e estejam parcialmente sobrepostos.

---

### Questão 2: Como a Mask R-CNN consegue separar cada instância, enquanto a segmentação semântica não consegue?

A razão fundamental reside na estratégia **"Detectar e Depois Segmentar" (Two-stage Top-Down)** da Mask R-CNN:

1. **Geração de Propostas de Região (RPN)**: A *Region Proposal Network* gera retângulos candidatos (*RoIs - Regions of Interest*) onde objetos individuais provavelmente estão localizados.
2. **Camada RoIAlign**: Para cada caixa proposta, a camada **RoIAlign** extrai o mapa de características exato correspondente àquela região específica usando **interpolação bilinear**. Isso evita erros de arredondamento espacial (presentes no antigo *RoIPool*), mantendo um alinhamento rigoroso pixel a pixel.
3. **Cabeça FCN Dedicada por RoI**: Para cada caixa desvinculada pelo RoIAlign, uma pequena sub-rede totalmente convolucional (FCN) prevê uma máscara binária de dimensão $m \times m$ (ex: $28 \times 28$) **exclusiva para o objeto dentro daquela caixa**.

Como o cálculo da máscara é isolado para cada proposta de região individual, objetos da mesma classe recebem máscaras independentes e cores distintas. A segmentação semântica tradicional, por outro lado, aplica uma convolução na imagem inteira e gera um único mapa de classes por pixel, sem o conceito de região proposta desvinculada.

---

### Questão 3: Qual a diferença de arquitetura entre Mask R-CNN e YOLOv8-Seg?

- **Mask R-CNN (Duas Etapas - Two-Stage)**:
  - *Funcionamento*: Extrai regiões candidatas (RPN) $\rightarrow$ ajusta alinhamento (RoIAlign) $\rightarrow$ gera máscara individual para cada RoI proposta.
  - *Vantagem*: Alta precisão e contornos extremamente detalhados.
  - *Desvantagem*: Maior custo computacional, inviabilizando aplicações de tempo real em dispositivos leves.

- **YOLOv8-Seg (Estágio Único - Single-Stage / Baseado no YOLACT)**:
  - *Funcionamento*: Utiliza um *Proto Module* que gera $k$ máscaras protótipo para a imagem inteira de forma paralela. Simultaneamente, a cabeça de detecção prevê coeficientes de máscara para cada caixa delimitadora. A máscara final de cada instância é a combinação linear dos protótipos recortada pelo retângulo da caixa delimitadora.
  - *Vantagem*: Altíssima velocidade (apropriado para tempo real e vídeo).
  - *Desvantagem*: A precisão do contorno depende do enquadramento exato da caixa delimitadora.

---

## 4. Justificativa Técnica da Alteração na Estratégia de Execução

No enunciado do **Desafio 2**, a instrução original solicitava:
> *"Siga as instruções para configurar o ambiente e executar o notebook de demonstração (`demo.ipynb`)"*

Embora tenhamos clonado o repositório oficial da Matterport ([`Mask_RCNN`](Mask_RCNN)) para estudo do código-fonte e dos arquivos originais, optamos por **adaptar a execução da inferência para a implementação oficial da Mask R-CNN disponível no PyTorch (`torchvision`)**. A seguir, detalham-se os motivos técnicos que fundamentam essa decisão:

### 1. Incompatibilidade Crítica de Versões (Depreciação do TensorFlow 1.x)
- O repositório da Matterport foi desenvolvido entre **2017 e 2018**, dependendo estritamente do **Python 3.4-3.6, TensorFlow 1.3-1.15 e Keras 2.0.8-2.2.4**.
- No ambiente de desenvolvimento moderno (Python 3.12), a versão `tensorflow==1.15` foi **descontinuada no PyPI** e não pode ser instalada nativamente.
- O notebook `demo.ipynb` original utiliza módulos como `tf.contrib` e `keras.engine.topology`, que foram **completamente removidos no TensorFlow 2.x**, disparando erros fatais de sintaxe e importação (`ModuleNotFoundError` e `AttributeError`).

### 2. Preservação do Conceito Matemático e Arquitetural (He et al., 2017)
- A Mask R-CNN não é uma biblioteca exclusiva, mas sim uma **arquitetura de Deep Learning**.
- A implementação `torchvision.models.detection.maskrcnn_resnet50_fpn` utiliza a **exata mesma arquitetura descrita no artigo original (He et al., 2017)**: *backbone* ResNet-50-FPN, *Region Proposal Network* (RPN), camada **RoIAlign** com interpolação bilinear e *FCN Head* para máscaras binárias por RoI.
- Ambas foram treinadas no mesmo conjunto de dados (**MS COCO**), garantindo rigor metodológico e resultados equivalentes.

### 3. Comparabilidade Direta e Reprodutibilidade
- A execução via script Python unificado ([`desafio2_detetive_instancias.py`](desafio2_detetive_instancias.py)) permitiu rodar a **Mask R-CNN** e o **YOLOv8-Seg** sob o mesmo pré-processamento, mesma imagem e mesma escala de pós-processamento.
- Isso viabilizou a geração de um gráfico comparativo direto (*side-by-side*) e um código 100% reprodutível em ambientes modernos sem a necessidade de configurar ambientes legados complexos ou contêineres obsoletos.

