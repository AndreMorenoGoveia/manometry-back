# manometry-back

API que recebe o laudo de uma manometria esofágica de alta resolução, recorta as
topografias de pressão das dez deglutições de água e devolve a classificação do
exame pela **Classificação de Chicago v4.0**, junto das métricas que sustentam a
decisão.

O modelo é o vencedor do estudo em
[`manometry-classification`](../manometry-classification): codificador
ConvNeXt-Base pré-treinado em ImageNet-22k sobre cada deglutição, agregação
`chicago_set` no nível do exame — um ramo que resume as posteriores por
deglutição junto das métricas de Chicago e um Set Transformer sobre os
embeddings. Partição estrita por paciente, oito classes.

```
laudo .docx/.rtf
   │  leitura dos PNG na ordem do documento
   ▼
paineis  ──▶ retrato = deglutição · paisagem = logotipo
   │
   │  unifier.images.clean_swallow  (código do construtor do dataset, verbatim)
   ▼
topografia recortada ──┬──▶ inversão da rampa jet ──▶ 24×241 ──▶ DCI, IRP4, DL, quebras…
                       │                                              │
                       └──▶ 224px ──▶ ConvNeXt-Base ──▶ embeddings ───┤
                                                                      ▼
                                            ramo Chicago + Set Transformer ──▶ 8 classes
```

## Por que este checkpoint e não o `fusion_heavy_deploy`

O checkpoint anteriormente marcado como *deploy* no repositório de
classificação foi treinado com as métricas de Chicago derivadas dos **CSV de
pressão** (`.cache/institutional_metrics.npy`). Um laudo não traz esses CSV, só
imagens — servi-lo aqui alimentaria o ramo Chicago com uma distribuição que ele
nunca viu.

O commit seguinte do repositório de classificação (`fdd46eb`) introduziu
`experiments/image_chicago.py`, que recupera as mesmas métricas exclusivamente
da imagem renderizada, e apontou `run_fusion_heavy.py` para elas — mas nenhum
modelo chegou a ser treinado nessa configuração. Este serviço usa o treino
`fusion_heavy_image_deploy`, que roda exatamente a mesma receita com as métricas
de imagem, de forma que **inferência e treino leem a mesma coisa**. O número de
desempenho que a API publica em `/api/model` é o desse treino, medido na sua
própria partição de teste.

`tests/test_inference_parity.py` reexecuta a partição de teste do treino pelo
caminho da API e exige que as probabilidades batam com as guardadas junto ao
checkpoint. É o que impede a transcrição da arquitetura de sair do lugar.

## Instalação

```bash
python3 -m pip install -r requirements.txt   # torch conforme a sua CUDA
```

O checkpoint e a calibração de cor não são versionados aqui: são artefatos
grandes e derivados de dados de paciente, e vivem em `manometry-classification`.
Com os dois repositórios lado a lado nada precisa ser configurado.

```bash
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
```

### Variáveis de ambiente

| variável | padrão | para quê |
|---|---|---|
| `MANOMETRY_CLASSIFICATION_ROOT` | `../manometry-classification` | raiz de onde saem checkpoint, calibração e corpus |
| `MANOMETRY_CHECKPOINT` | `<root>/artifacts/leakage_fusion/fusion_heavy_image_deploy/…/model.pt` | outro checkpoint |
| `MANOMETRY_RESULT_JSON` | `result.json` ao lado do checkpoint | métricas publicadas em `/api/model` |
| `MANOMETRY_CALIBRATION` | `<root>/.cache/image_chicago_calibration.json` | calibração da rampa de cor |
| `MANOMETRY_DATASET_ROOT` | `<root>/data` | corpus, usado pelas imagens de referência e pelos laudos de exemplo |
| `MANOMETRY_REFERENCE_DIR` | `assets/reference` | PNG de referência já exportados |
| `MANOMETRY_DEVICE` | `auto` | `cuda`, `cpu` |
| `MANOMETRY_LAZY_MODEL` | `true` | `false` carrega o checkpoint no boot, para readiness probe |
| `MANOMETRY_MAX_UPLOAD_BYTES` | `67108864` | limite do upload |
| `MANOMETRY_ALLOWED_ORIGINS` | `localhost:5173,4173` | CORS |

## Endpoints

| método | rota | o que faz |
|---|---|---|
| `POST` | `/api/analyze` | laudo `.docx`/`.rtf` → diagnóstico, ranking, métricas por deglutição, portas de Chicago |
| `GET` | `/api/health` | estado do serviço e se o checkpoint está acessível |
| `GET` | `/api/model` | procedência do modelo e desempenho na partição de teste |
| `GET` | `/api/diseases` | texto clínico das oito classes |
| `GET` | `/api/diseases/{slug}/reference.png` | topografia real da classe, da partição de teste |
| `GET` | `/api/vocabulary` | rótulos dos padrões de deglutição e das métricas |
| `GET` | `/api/samples` | classes para as quais há laudo de exemplo nesta máquina |
| `GET` | `/api/sample-report?diagnosis=<slug>` | gera um laudo `.docx` de demonstração a partir do corpus |

O arquivo enviado nunca toca o disco: é lido em memória, as imagens são
recortadas, medidas e descartadas com a resposta.

## Scripts

```bash
scripts/sync_vendor.sh                  # re-copia os módulos de origem (ver app/vendor/README.md)
python3 scripts/export_reference_images.py   # escolhe e exporta a topografia de cada classe
python3 scripts/make_sample_docx.py --class achalasia_type_ii --out demo.docx
```

## Testes

```bash
python3 -m pytest              # rápido: leitor, recorte, texto clínico, HTTP
python3 -m pytest --runslow    # inclui a paridade contra o treino (carrega o checkpoint)
```

## Limites que este serviço não esconde

- **As métricas são proxies de imagem.** DCI, IRP, latência e quebras são
  recuperadas invertendo a rampa de cor do gráfico, ancorada numa calibração
  global ajustada apenas em exames normais de treino. Servem para justificar a
  decisão; não são medidas do sistema de aquisição e não devem ser transcritas
  para um laudo.
- **Duas classes da v4.0 estão fora**: obstrução da saída da JEG (n=12) e
  acalásia tipo III (n=5) não têm exames suficientes para uma partição em três
  partes. Um exame dessas condições será forçado à classe mais próxima.
- **`hypotensive_egj` não é um diagnóstico da v4.0**, e sim o rótulo que o
  serviço de origem usa para barreira antirrefluxo incompetente com corpo
  esofágico normal. A ressalva viaja junto do texto clínico da classe.
- **As imagens do corpus trazem as anotações de medida do próprio software**
  (caixas, distâncias em cm, tempos em s e valores de DCI desenhados sobre o
  traçado). Elas estão presentes tanto no treino quanto no teste, então parte do
  desempenho relatado pode vir de o codificador aprender a lê-las em vez de ler
  a topografia. Remover essas anotações e repetir a medição é trabalho para o
  repositório de classificação, não para este.
- **Contexto clínico não entra.** A v4.0 exige sintomas para nomear esôfago
  hipercontrátil e espasmo distal como distúrbio; uma imagem não os carrega.

Ferramenta de pesquisa e apoio à decisão. Não substitui a leitura do exame por
um médico habilitado.
