# Visão Computacional - Segmentação Semântica e de Instâncias

Este repositório contém o projeto e os códigos práticos desenvolvidos para a aula de **Visão Computacional** da pós-graduação. O projeto aborda a transição das caixas delimitadoras (*bounding boxes*) para o entendimento pixel a pixel através da **Segmentação de Instâncias**, focando na implementação e comparação entre a **Mask R-CNN** e o **YOLOv8-Seg**.

---

## 📁 Estrutura do Repositório

```text
vc-projeto-final/
├── 6. A Precisão do Pixel - Segmentação Semântica e de Instâncias.ipynb - Colab.pdf  # Material didático e enunciados
├── Mask_RCNN/                             # Repositório clonado da Matterport (referência histórica)
├── desafio2_detetive_instancias.py        # Script Python principal de inferência comparativa
├── mask_rcnn_vs_yolov8.png                # Imagem gerada com a comparação dos modelos lado a lado
├── relatorio_desafio2.md                  # Relatório técnico completo com as respostas do Desafio 2
└── README.md                              # Instruções de reprodução do ambiente e execução
```

---

## 🛠️ Requisitos do Sistema

- **Python**: 3.10 ou superior (testado e validado no Python 3.12)
- **Git**
- **Acesso à Internet** (para download automático dos pesos dos modelos pré-treinados no COCO)

---

## 🚀 Passo a Passo para Configuração e Execução

### 1. Clonar o Repositório

Abra o terminal e clone este repositório para a sua máquina local:

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd vc-projeto-final
```

---

### 2. Clonar o Repositório da Matterport (Mask R-CNN)

Para manter a estrutura completa do Desafio 2, clone o repositório original da Matterport dentro da pasta do projeto (caso não tenha vindo no clone inicial):

```bash
git clone https://github.com/matterport/Mask_RCNN.git
```

---

### 3. Instalar as Dependências

Recomenda-se criar um ambiente virtual (`venv`) antes de instalar os pacotes.

#### No Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### No Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Instale o PyTorch (versão CPU ou GPU) e as demais bibliotecas necessárias:

```bash
# Instalação do PyTorch e Torchvision (versão CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalação do YOLOv8 e ferramentas de visualização
pip install ultralytics opencv-python matplotlib pillow requests polars
```

---

### 4. Executar o Script do Desafio 2

Execute o script `desafio2_detetive_instancias.py`:

```bash
python desafio2_detetive_instancias.py
```

---

## 📊 O Que Acontece ao Executar?

1. **Download da Imagem de Teste**: O script baixará automaticamente uma imagem de teste real contendo múltiplos objetos sobrepostos (pedestres e um ônibus).
2. **Inferência da Mask R-CNN**: Carrega a rede **Mask R-CNN (ResNet-50-FPN)** pré-treinada do `torchvision`, calcula as máscaras binárias individuais via **RoIAlign** e desenha caixas e rótulos.
3. **Inferência do YOLOv8-Seg**: Carrega o modelo **YOLOv8-Seg (`yolov8n-seg.pt`)** da Ultralytics e calcula a segmentação de instâncias em estágio único.
4. **Geração do Painel Comparativo**: Salva a imagem `mask_rcnn_vs_yolov8.png` na raiz do projeto com a comparação visual dos três painéis (Original, Mask R-CNN e YOLOv8-Seg).

---

## 📖 Relatório Técnico

Para conferir a fundamentação teórica completa, as arquiteturas detalhadas (RPN, RoIAlign, FCN Head vs Protótipos YOLOv8), as respostas conceituais do Desafio 2 e a **justificativa técnica detalhada sobre a adaptação do ambiente (PyTorch vs TF 1.15 legado)**, consulte o arquivo [`relatorio_desafio2.md`](relatorio_desafio2.md).

